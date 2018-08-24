// Check that we test WF conditions for fn arguments. Because the
// current code is so goofy, this is only a warning for now.

#![feature(rustc_attrs)]
#![allow(dead_code)]
#![allow(unused_variables)]

trait Bar<T:Eq+?Sized> { }

trait Foo {
    fn bar<A>(&self) where A: Bar<Self> {
        //~^ ERROR E0277
        //
        // Here, Eq ought to be implemented.
    }
}

#[rustc_error]
fn main() { }
