//! regression test for https://github.com/rust-lang/rust/issues/21177
trait Trait {
    type A;
    type B;
}

fn foo<T: Trait<A = T::B>>() {}
//~^ ERROR cycle detected

fn main() {}
