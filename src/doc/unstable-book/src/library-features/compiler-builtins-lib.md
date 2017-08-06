# `compiler_builtins_lib`

The tracking issue for this feature is: None.

------------------------

This feature is required to link to the `compiler_builtins` crate which contains
"compiler intrinsics". Compiler intrinsics are software implementations of basic
operations like multiplication of `u64`s. These intrinsics are only required on
platforms where these operations don't directly map to a hardware instruction.

You should never need to explicitly link to the `compiler_builtins` crate when
building "std" programs as `compiler_builtins` is already in the dependency
graph of `std`. But you may need it when building `no_std` **binary** crates. If
you get a *linker* error like:

``` text
$PWD/src/main.rs:11: undefined reference to `__aeabi_lmul'
$PWD/src/main.rs:11: undefined reference to `__aeabi_uldivmod'
```

That means that you need to link to this crate.

When you link to this crate, make sure it only appears once in your crate
dependency graph. Also, it doesn't matter where in the dependency graph you
place the `compiler_builtins` crate.

<!-- NOTE(ignore) doctests don't support `no_std` binaries -->

``` rust,ignore
#![feature(compiler_builtins_lib)]
#![no_std]

extern crate compiler_builtins;
```
