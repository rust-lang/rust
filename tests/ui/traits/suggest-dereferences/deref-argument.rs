//@ run-rustfix
//! diagnostic test for #90997.
//! test that E0277 suggests dereferences to satisfy bounds when the referent is `Copy` or boxed.
use std::ops::Deref;

trait Test {
    fn test(self);
}
fn consume_test(x: impl Test) { x.test() }

impl Test for u32 {
    fn test(self) {}
}
struct MyRef(u32);
impl Deref for MyRef {
    type Target = u32;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct NonCopy;
impl Test for NonCopy {
    fn test(self) {}
}

fn main() {
    let my_ref = MyRef(0);
    consume_test(my_ref);
    //~^ ERROR the trait bound `MyRef: Test` is not satisfied
    //~| SUGGESTION *

    let nested_box = Box::new(Box::new(Box::new(NonCopy)));
    consume_test(nested_box);
    //~^ ERROR the trait bound `Box<Box<Box<NonCopy>>>: Test` is not satisfied
    //~| SUGGESTION ***
}
