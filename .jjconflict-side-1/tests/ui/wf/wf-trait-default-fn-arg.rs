// Check that we test WF conditions for fn arguments. Because the
// current code is so goofy, this is only a warning for now.


#![allow(dead_code)]
#![allow(unused_variables)]

struct Bar<T:Eq+?Sized> { value: Box<T> }

trait Foo {
    fn bar(&self, x: &Bar<Self>) {
        //~^ ERROR E0277
        //
        // Here, Eq ought to be implemented.
    }
}


fn main() { }
