#### Note: this error code is no longer emitted by the compiler.

The argument to the `llvm_asm` macro is not well-formed.

Erroneous code example:

```ignore (no longer emitted)
llvm_asm!("nop" "nop");
```
