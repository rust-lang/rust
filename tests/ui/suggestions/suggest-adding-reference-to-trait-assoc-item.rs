//@ run-rustfix
#![allow(unused_variables)]

fn foo(foo: &mut usize) {
    todo!()
}

fn bar(bar: &usize) {
    todo!()
}

fn main() {
    foo(Default::default()); //~ ERROR the trait bound `&mut usize: Default` is not satisfied
    bar(Default::default()); //~ ERROR the trait bound `&usize: Default` is not satisfied
}
