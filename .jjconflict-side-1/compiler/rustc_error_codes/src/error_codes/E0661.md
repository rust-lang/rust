#### Note: this error code is no longer emitted by the compiler.

An invalid syntax was passed to the second argument of an `llvm_asm` macro line.

Erroneous code example:

```ignore (no longer emitted)
let a;
llvm_asm!("nop" : "r"(a));
```
