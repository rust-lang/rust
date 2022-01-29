use std::marker;

pub struct TypeWithState<State>(marker::PhantomData<State>);
pub struct MyState;

pub fn foo<State>(_: TypeWithState<State>) {}

pub fn bar() {
   foo(TypeWithState(marker::PhantomData));
   //~^ ERROR type annotations needed [E0282]
}

fn main() {
}
