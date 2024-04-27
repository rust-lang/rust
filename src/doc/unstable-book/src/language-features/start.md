# `start`

The tracking issue for this feature is: [#29633]

[#29633]: https://github.com/rust-lang/rust/issues/29633

------------------------

Allows you to mark a function as the entry point of the executable, which is
necessary in `#![no_std]` environments.

The function marked `#[start]` is passed the command line parameters in the same
format as the C main function (aside from the integer types being used).
It has to be non-generic and have the following signature:

```rust,ignore (only-for-syntax-highlight)
# let _:
fn(isize, *const *const u8) -> isize
# ;
```

This feature should not be confused with the `start` *lang item* which is
defined by the `std` crate and is written `#[lang = "start"]`.

## Usage together with the `std` crate

`#[start]` can be used in combination with the `std` crate, in which case the
normal `main` function (which would get called from the `std` crate) won't be
used as an entry point.
The initialization code in `std` will be skipped this way.

Example:

```rust
#![feature(start)]

#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    0
}
```

Unwinding the stack past the `#[start]` function is currently considered
Undefined Behavior (for any unwinding implementation):

```rust,ignore (UB)
#![feature(start)]

#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    std::panic::catch_unwind(|| {
        panic!(); // panic safely gets caught or safely aborts execution
    });

    panic!(); // UB!

    0
}
```
