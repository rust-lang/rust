#### Note: this error code is no longer emitted by the compiler.

An invalid input operand constraint was passed to the `llvm_asm` macro
(third line).

Erroneous code example:

```ignore (no longer emitted)
llvm_asm!("xor %eax, %eax"
          :
          : "=test"("a")
         );
```
