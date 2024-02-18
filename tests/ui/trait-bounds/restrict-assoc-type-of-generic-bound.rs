//@ run-rustfix
pub trait MyTrait {
    type T;

    fn bar(self) -> Self::T;
}

pub fn foo<A: MyTrait, B>(a: A) -> B {
    return a.bar(); //~ ERROR mismatched types
}
fn main() {}
