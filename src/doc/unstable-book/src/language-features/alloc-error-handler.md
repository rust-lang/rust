# `alloc_error_handler`

The tracking issue for this feature is: [#51540]

[#51540]: https://github.com/rust-lang/rust/issues/51540

------------------------

This attribute is mandatory when using the alloc crate without the std crate. It is used when handling out of memory (OOM) allocation error, and is called
by `alloc::alloc::handle_alloc_error`

``` rust,ignore (partial-example)
#![feature(alloc_error_handler)]
#![no_std]

#[alloc_error_handler]
fn foo(_: core::alloc::Layout) -> ! {
    // …
}
```
