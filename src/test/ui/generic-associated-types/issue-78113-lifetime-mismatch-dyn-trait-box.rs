// Test for diagnostics when we have mismatched lifetime due to implict 'static lifetime in GATs

// check-fail

#![feature(generic_associated_types)]

pub trait A {}
impl A for &dyn A {}
impl A for Box<dyn A> {}

pub trait B {
    type T<'a>: A;
}

impl B for () {
    type T<'a> = Box<dyn A + 'a>; //~ incompatible lifetime on type
}

fn main() {}
