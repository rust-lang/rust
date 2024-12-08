#![allow(unused_variables, dead_code)]
//@no-rustfix
fn takes_an_immutable_reference(a: &i32) {}
fn takes_a_mutable_reference(a: &mut i32) {}

mod issue11268 {
    macro_rules! x {
        ($f:expr) => {
            $f(&mut 1);
        };
    }

    fn f() {
        x!(super::takes_an_immutable_reference);
        x!(super::takes_a_mutable_reference);
    }
}

struct MyStruct;

impl MyStruct {
    fn takes_an_immutable_reference(&self, a: &i32) {}

    fn takes_a_mutable_reference(&self, a: &mut i32) {}
}

#[warn(clippy::unnecessary_mut_passed)]
fn main() {
    // Functions
    takes_an_immutable_reference(&mut 42);
    //~^ ERROR: the function `takes_an_immutable_reference` doesn't need a mutable referen
    //~| NOTE: `-D clippy::unnecessary-mut-passed` implied by `-D warnings`
    let as_ptr: fn(&i32) = takes_an_immutable_reference;
    as_ptr(&mut 42);
    //~^ ERROR: the function `as_ptr` doesn't need a mutable reference

    // Methods
    let my_struct = MyStruct;
    my_struct.takes_an_immutable_reference(&mut 42);
    //~^ ERROR: the method `takes_an_immutable_reference` doesn't need a mutable reference

    // No error

    // Functions
    takes_an_immutable_reference(&42);
    let as_ptr: fn(&i32) = takes_an_immutable_reference;
    as_ptr(&42);

    takes_a_mutable_reference(&mut 42);
    let as_ptr: fn(&mut i32) = takes_a_mutable_reference;
    as_ptr(&mut 42);

    let a = &mut 42;
    takes_an_immutable_reference(a);

    // Methods
    my_struct.takes_an_immutable_reference(&42);
    my_struct.takes_a_mutable_reference(&mut 42);
    my_struct.takes_an_immutable_reference(a);
}
