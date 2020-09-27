An unknown intrinsic function was declared.

Erroneous code example:

```compile_fail,E0093
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn foo(); // error: unrecognized intrinsic function: `foo`
}

fn main() {
    unsafe {
        foo();
    }
}
```

Please check you didn't make a mistake in the function's name. All intrinsic
functions are defined in `compiler/rustc_codegen_llvm/src/intrinsic.rs` and in
`library/core/src/intrinsics.rs` in the Rust source code. Example:

```
#![feature(intrinsics)]

extern "rust-intrinsic" {
    fn atomic_fence(); // ok!
}

fn main() {
    unsafe {
        atomic_fence();
    }
}
```
