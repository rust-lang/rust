#![feature(marker_trait_attr)]

#[marker]
trait Marker {
    const N: usize = 0;
    fn do_something() {}
}

struct OverrideConst;
impl Marker for OverrideConst {
//~^ ERROR impls for marker traits cannot contain items
    const N: usize = 1;
}

struct OverrideFn;
impl Marker for OverrideFn {
//~^ ERROR impls for marker traits cannot contain items
    fn do_something() {
        println!("Hello world!");
    }
}

fn main() {}
