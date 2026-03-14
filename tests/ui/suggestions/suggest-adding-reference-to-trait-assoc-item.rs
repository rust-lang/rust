//@ run-rustfix
#![allow(unused_variables, todo_macro_uses)]

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
