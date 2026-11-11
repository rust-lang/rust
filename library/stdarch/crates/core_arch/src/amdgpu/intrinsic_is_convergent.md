This intrinsic does not behave like a normal function call; it is a "[convergent]" operation and as such has non-standard control-flow effects which need special treatment by the language.
Rust currently does not properly support convergent operations.
This operation is hence provided on a best-effort basis.
Using it may result in incorrect code under some circumstances.

[convergent]: https://llvm.org/docs/ConvergentOperations.html
