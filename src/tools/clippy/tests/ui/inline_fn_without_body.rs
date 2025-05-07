#![warn(clippy::inline_fn_without_body)]
#![allow(clippy::inline_always)]

trait Foo {
    #[inline]
    //~^ inline_fn_without_body
    fn default_inline();

    #[inline(always)]
    //~^ inline_fn_without_body
    fn always_inline();

    #[inline(never)]
    //~^ inline_fn_without_body
    fn never_inline();

    #[inline]
    fn has_body() {}
}

fn main() {}
