% No stdlib

Rust’s standard library provides a lot of useful functionality, but assumes
support for various features of its host system: threads, networking, heap
allocation, and others. There are systems that do not have these features,
however, and Rust can work with those too! To do so, we tell Rust that we
don’t want to use the standard library via an attribute: `#![no_std]`.

> Note: This feature is technically stable, but there are some caveats. For
> one, you can build a `#![no_std]` _library_ on stable, but not a _binary_.
> For details on libraries without the standard library, see [the chapter on
> `#![no_std]`](using-rust-without-the-standard-library.html)

Obviously there's more to life than just libraries: one can use
`#[no_std]` with an executable, controlling the entry point is
possible in two ways: the `#[start]` attribute, or overriding the
default shim for the C `main` function with your own.

The function marked `#[start]` is passed the command line parameters
in the same format as C:

```rust
# #![feature(libc)]
#![feature(lang_items)]
#![feature(start)]
#![no_std]

// Pull in the system libc library for what crt0.o likely requires
extern crate libc;

// Entry point for this program
#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    0
}

// These functions and traits are used by the compiler, but not
// for a bare-bones hello world. These are normally
// provided by libstd.
#[lang = "eh_personality"] extern fn eh_personality() {}
#[lang = "panic_fmt"] fn panic_fmt() -> ! { loop {} }
# #[lang = "eh_unwind_resume"] extern fn rust_eh_unwind_resume() {}
# #[no_mangle] pub extern fn rust_eh_register_frames () {}
# #[no_mangle] pub extern fn rust_eh_unregister_frames () {}
# // fn main() {} tricked you, rustdoc!
```

To override the compiler-inserted `main` shim, one has to disable it
with `#![no_main]` and then create the appropriate symbol with the
correct ABI and the correct name, which requires overriding the
compiler's name mangling too:

```rust
# #![feature(libc)]
#![feature(lang_items)]
#![feature(start)]
#![no_std]
#![no_main]

extern crate libc;

#[no_mangle] // ensure that this symbol is called `main` in the output
pub extern fn main(argc: i32, argv: *const *const u8) -> i32 {
    0
}

#[lang = "eh_personality"] extern fn eh_personality() {}
#[lang = "panic_fmt"] fn panic_fmt() -> ! { loop {} }
# #[lang = "eh_unwind_resume"] extern fn rust_eh_unwind_resume() {}
# #[no_mangle] pub extern fn rust_eh_register_frames () {}
# #[no_mangle] pub extern fn rust_eh_unregister_frames () {}
# // fn main() {} tricked you, rustdoc!
```


The compiler currently makes a few assumptions about symbols which are available
in the executable to call. Normally these functions are provided by the standard
library, but without it you must define your own.

The first of these two functions, `eh_personality`, is used by the
failure mechanisms of the compiler. This is often mapped to GCC's
personality function (see the
[libstd implementation](../std/rt/unwind/index.html) for more
information), but crates which do not trigger a panic can be assured
that this function is never called. The second function, `panic_fmt`, is
also used by the failure mechanisms of the compiler.
