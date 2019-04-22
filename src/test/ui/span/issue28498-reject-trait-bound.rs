// Demonstrate that having a trait bound causes dropck to reject code
// that might indirectly access previously dropped value.
//
// Compare with run-pass/issue28498-ugeh-with-trait-bound.rs

use std::fmt;

#[derive(Debug)]
struct ScribbleOnDrop(String);

impl Drop for ScribbleOnDrop {
    fn drop(&mut self) {
        self.0 = format!("DROPPED");
    }
}

struct Foo<T:fmt::Debug>(u32, T);

impl<T:fmt::Debug> Drop for Foo<T> {
    fn drop(&mut self) {
        // Use of `unsafe_destructor_blind_to_params` is unsound,
        // because we access `T` fmt method when we pass `self.1`
        // below, and thus potentially read from borrowed data.
        println!("Dropping Foo({}, {:?})", self.0, self.1);
    }
}

fn main() {
    let (last_dropped, foo0);
    let (foo1, first_dropped);

    last_dropped = ScribbleOnDrop(format!("last"));
    first_dropped = ScribbleOnDrop(format!("first"));
    foo0 = Foo(0, &last_dropped); // OK
    foo1 = Foo(1, &first_dropped);
    //~^ ERROR `first_dropped` does not live long enough

    println!("foo0.1: {:?} foo1.1: {:?}", foo0.1, foo1.1);
}
