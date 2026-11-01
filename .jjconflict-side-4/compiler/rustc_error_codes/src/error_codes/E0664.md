#### Note: this error code is no longer emitted by the compiler.

A clobber was surrounded by braces in the `llvm_asm` macro.

Erroneous code example:

```ignore (no longer emitted)
llvm_asm!("mov $$0x200, %eax"
          :
          :
          : "{eax}"
         );
```
