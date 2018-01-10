The project is in its early stages: contributions are welcome and
would be **very** helpful, but the project is not *yet* optimized for
contributors. Moreover, it is doubly experimental, so there's no
guarantee that any work here would reach production. That said, here
are some arias where contributions would be **especially** welcome:


* Designing internal data structures: RFC only outlines the
  constraints, it's an open question how to satisfy them in the
  optimal way. See `ARCHITECTURE.md` for current design questions.
  
* Porting libsyntax parser to libsyntax2: currently libsyntax2 parses
  only a tiny subset of Rust. This should be fixed by porting parsing
  functions from libsyntax one by one.
  
* Writing validators: by design, libsyntax2 is very lax about the
  input. For example, the lexer happily accepts unclosed strings. The
  idea is that there should be a higher level visitor, which walks the
  syntax tree after parsing and produces all the warnings. Alas,
  there's no such visitor yet :( Would you like to write one? :)
  
* Creating tests: it would be tremendously helpful to read each of
  libsyntax and libsyntax2 parser functions and crate a small separate
  test cases to cover each and every edge case.
  
* Building stuff with libsyntax2: it would be really cool to compile
  libsyntax2 to WASM and add *client side* syntax validation to rust
  playground!


Do take a look at the issue tracker, and try to read other docs in
this folder.
