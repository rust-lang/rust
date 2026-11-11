#### Note: this error code is no longer emitted by the compiler.

Cannot convert inline assembly operand to a single LLVM value.

Erroneous code example:

```ignore (no longer emitted)
#![feature(llvm_asm)]

fn main() {
    unsafe {
        llvm_asm!("" :: "r"("")); // error!
    }
}
```

This error usually happens when trying to pass in a value to an input inline
assembly operand that is actually a pair of values. In particular, this can
happen when trying to pass in a slice, for instance a `&str`. In Rust, these
values are represented internally as a pair of values, the pointer and its
length. When passed as an input operand, this pair of values can not be
coerced into a register and thus we must fail with an error.
