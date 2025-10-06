// The duplicate macro will create a copy of the item with the given identifier.

//@ check-pass
//@ proc-macro: duplicate.rs
//@ ignore-backends: gcc

#[macro_use]
extern crate duplicate;

#[duplicate(MyCopy)]
struct MyStruct {
    number: i32,
}

trait TestTrait {
    #[duplicate(TestType2)]
    type TestType;

    #[duplicate(required_fn2)]
    fn required_fn(&self);

    #[duplicate(provided_fn2)]
    fn provided_fn(&self) {}
}

impl TestTrait for MyStruct {
    #[duplicate(TestType2)]
    type TestType = f64;

    #[duplicate(required_fn2)]
    fn required_fn(&self) {}
}

fn main() {
    let s = MyStruct { number: 42 };
    s.required_fn();
    s.required_fn2();
    s.provided_fn();
    s.provided_fn2();

    let s = MyCopy { number: 42 };
}
