//@ known-bug: #112905
//@ check-pass

// Classified as an issue with implied bounds:
// https://github.com/rust-lang/rust/issues/112905#issuecomment-1757847998

#![forbid(unsafe_code)] // No `unsafe!`
#![feature(type_alias_impl_trait)]

use std::any::Any;

/// Anything covariant will do, for this demo.
type T<'lt> = &'lt str;

type F<'a, 'b> = impl 'static + Fn(T<'a>) -> T<'b>;

#[define_opaque(F)]
fn helper<'a, 'b>(_: [&'b &'a (); 0]) -> F<'a, 'b> {
    |x: T<'a>| -> T<'b> { x } // this should *not* be `: 'static`
}

fn exploit<'a, 'b>(a: T<'a>) -> T<'b> {
    let f: F<'a, 'a> = helper([]);
    let any = Box::new(f) as Box<dyn Any>;

    let f: F<'a, 'static> = *any.downcast().unwrap_or_else(|_| unreachable!());

    f(a)
}

fn main() {
    let r: T<'static> = {
        let local = String::from("...");
        exploit(&local)
    };
    // Since `r` now dangles, we can easily make the use-after-free
    // point to newly allocated memory!
    let _unrelated = String::from("UAF");
    dbg!(r); // may print `UAF`! Run with `miri` to see the UB.
}
