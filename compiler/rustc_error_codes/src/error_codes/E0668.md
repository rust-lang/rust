#### Note: this error code is no longer emitted by the compiler.

Malformed inline assembly rejected by LLVM.

Erroneous code example:

```ignore (no longer emitted)
#![feature(llvm_asm)]

fn main() {
    let rax: u64;
    unsafe {
        llvm_asm!("" :"={rax"(rax));
        println!("Accumulator is: {}", rax);
    }
}
```

LLVM checks the validity of the constraints and the assembly string passed to
it. This error implies that LLVM seems something wrong with the inline
assembly call.

In particular, it can happen if you forgot the closing bracket of a register
constraint (see issue #51430), like in the previous code example.
