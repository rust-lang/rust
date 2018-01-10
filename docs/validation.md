Fixmes:

Lexer:
* Fix `is_whitespace`, add more tests
* Add more thorough tests for idents for XID_Start & XID_Continue
* Validate that float and integer literals use digits only of the appropriate
  base, and are in range
* Validation for unclosed char literal
* Strings are completely wrong: more tests and comparison with libsyntax.
* Comment lexing is completely wrong

Parser:
* Figure out what is the expected state of attribute grammar.
  Token trees or something more structured? Token trees would be unfortunate: no extend selection =/