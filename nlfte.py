import regex2dfa
import fte.encoder
import fte
import fte.bit_ops
import math
import inference
import aux_regex


# Given a list of auxiliary verb sentences, infer and return the k-reversible FSM underlying it
# Args:
# param1 (list): list of strings, where each string is an auxiliary verb sentence
# param2 (int): the k in k-reversible inference; default is 2 which is applicable for auxiliary verbs sentences,
# and noun phrases
# Returns:
# fst (openfst): open fst object representing the FSM underlying the input list of strings
def infer(S, k=2):
    a = inference.minimize(inference.k_RI(S, k), inference.get_alphabet(S))
    return a.to_fsm()



# NlfteObj allows encryption and decryption from specified format input
#    Args:
#        regex (str): the regex to be input to the format; if None then use fsm directly
#        fsm (openfst): the fst inferred from the inference algorithm
#        fixed_slice (int): initial maximum length of strings considered in the language
class nlfteObj:
    def __init__(self, regex=None, fsm=None, fixed_slice=5):
        self.fixed_slice = fixed_slice
        self.regex = regex
        self.fsm = fsm
        self.fteObj = self.get_fte_obj()

    # Return fte object from regex, or fsm
    # Note: if both regex and fsm are defined, regex is prioritized
    def get_fte_obj(self):
        if self.regex is None and self.fsm is None:
            print("Error: No specified format.")
            return

        if self.regex is not None:
            # get fsm from regex
            self.fsm = regex2dfa.regex2dfa(self.regex)

        # Get FTE Object + increase fixed slice until language is expressive enough
        while True:
            try:
                fteObj = fte.encoder.DfaEncoder(self.fsm, self.fixed_slice)
            except Exception as e:
                self.fixed_slice += 5
            else:
                break
        return fteObj

    # Encrypt a plaintext into a specified format
    # Args:
    #   plaintext (str): string to encrypt
    # Returns:
    #   cipher (str): Obfuscated cipher in the specified format
    def encrypt(self, plaintext):

        # dummy plaintext with same length as actual plaintext to determine length of AES cipher
        dummy_p = fte.bit_ops.random_bytes(len(plaintext))
        aes_dummy_cipher = self.fteObj._encrypter.encrypt(dummy_p)

        # get capacity of the language
        capacity = int(math.floor(self.fteObj.getCapacity() / 8.0))

        # get random bytes length
        random_bytes_len = len(aes_dummy_cipher) - (
                    capacity - fte.encoder.DfaEncoderObject._COVERTEXT_HEADER_LEN_CIPHERTTEXT)

        # if random bytes len greater than 0 create a new fteObj with fixed slice increased by 8*random_bytes_len
        while random_bytes_len > 0:
            # increse fixed slice and create new Fte Object
            self.fixed_slice += 8 * random_bytes_len
            self.fteObj = fte.encoder.DfaEncoder(self.fsm, self.fixed_slice)
            # get capacity of the language
            capacity = int(math.floor(self.fteObj.getCapacity() / 8.0))
            # get random bytes length
            random_bytes_len = len(aes_dummy_cipher) - (
                        capacity - fte.encoder.DfaEncoderObject._COVERTEXT_HEADER_LEN_CIPHERTTEXT)

        return self.fteObj.encode(plaintext)

    # Decrypt a cipher text from specified format
    # Args:
    #   cipher (str): obfuscated input cipher string to decrypt
    # Returns:
    #   plaintext (str): orriginal plaintext
    def decrypt(self, cipher):
        [plaintext, remainder] = self.fteObj.decode(cipher)
        return plaintext