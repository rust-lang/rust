#![deny(unused_attributes)]

trait Trait {
    #[inline] //~ ERROR `#[inline]` is ignored on function prototypes
    fn foo();
}

extern "C" {
    #[inline] //~ ERROR `#[inline]` is ignored on function prototypes
    fn foo();
}

fn main() {}
