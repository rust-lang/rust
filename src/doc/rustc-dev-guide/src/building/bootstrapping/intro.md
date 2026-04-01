# Bootstrapping the compiler

[*Bootstrapping*][boot] is the process of using a compiler to compile itself.
More accurately, it means using an older compiler to compile a newer version
of the same compiler.

This raises a chicken-and-egg paradox: where did the first compiler come from?
It must have been written in a different language. In Rust's case it was
[written in OCaml][ocaml-compiler]. However, it was abandoned long ago, and the
only way to build a modern version of rustc is with a slightly less modern
version.

This is exactly how `x.py` works: it downloads the current beta release of
rustc, then uses it to compile the new compiler.

In this section, we give a high-level overview of
[what Bootstrap does](./what-bootstrapping-does.md), followed by a high-level
introduction to [how Bootstrap does it](./how-bootstrap-does-it.md).

Additionally, see [debugging bootstrap](./debugging-bootstrap.md) to learn
about debugging methods.

[boot]: https://en.wikipedia.org/wiki/Bootstrapping_(compilers)
[ocaml-compiler]: https://github.com/rust-lang/rust/tree/ef75860a0a72f79f97216f8aaa5b388d98da6480/src/boot
