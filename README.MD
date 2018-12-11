# NLFTE (Natural Language Format Transforming Encryption)

NLFTE is a python library that provides the tools to obfuscating ciphers in the form of natural language and more specifically auxiliary verb sentences in the form Subject-Auxiliary Verb Phrase-Object

## Installation 
1. Install openFST following instructions at http://www.openfst.org/twiki/bin/view/FST/DistInstall
2. Install fst a python interface for openFST 
```
pip install pyfst
```


## Example Usage

```python
import nlfte
import aux_regex

### Initialize nlfte Object with regular expression ####
r = aux_regex.r1 
obj = nlfte.nlfteObj(regex=r)

obj.encrypt(plaintext)  
obj.decrypt(cipher) 

#### Initialize nlfte Object with fsm from a list of auxiliary verb sentences S ####

fsm = nlfte.infer(S, k) # Given a list of auxiliary verb sentences, infer and return the k-reversible FSM underlying it

nlfteObj = nlfteObj(fsm=fsm)
obj.encrypt(plaintext)  
obj.decrypt(cipher) 
``` 

## Notes

The aux_regex module contains regular expressions for auxiliary verb sentences derived with the nlfte.infer(S, k)
method which implements an inference algorithm for k-reversible languages [1]. 
Note that for auxiliary verb sentences k = 2 [2]. <br />
For more information on this choice refer to the project report. 
Even more information can be found in the references bellow:<br />
[1] Angluin, Dana. "Inference of reversible languages." Journal of the ACM (JACM) 29.3 (1982): 741-765.<br />
[2] Berwick, Robert C., and Sam Pilato. "Learning syntax by automata induction." Machine learning 2.1 (1987): 9-38.


