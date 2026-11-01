#![feature(return_type_notation)]
#![deny(refining_impl_trait)]

trait Trait {
    fn f() -> impl Sized;
}

impl Trait for () {
    fn f() {}
    //~^ ERROR impl trait in impl method signature does not match trait method signature
}

fn main() {}
