//@ known-bug: rust-lang/rust#124833
#![feature(generic_const_items)]

trait Trait {
    const C<'a>: &'a str;
}

impl Trait for () {
    const C<'a>:  = "C";
}
