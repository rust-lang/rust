// revisions: cfail
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub struct Ref<'a>(&'a i32);

impl<'a> Ref<'a> {
    pub fn foo<const A: usize>() -> [(); A - 0] {
        Self::foo()
        //~^ error: type annotations needed
    }
}

fn main() {}
