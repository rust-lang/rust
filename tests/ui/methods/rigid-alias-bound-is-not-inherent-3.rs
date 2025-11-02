use std::ops::Deref;

trait Trait1 {
    fn call_me(&self) {}
}

impl<T> Trait1 for T {}

trait Trait2 {
    fn call_me(&self) {}
}

impl<T> Trait2 for T {}

pub fn foo<T, U>(x: T)
where
    T: Deref<Target = U>,
    U: Trait1,
{
    // This should be ambiguous. The fact that there's an inherent where-bound
    // candidate for `U` should not impact the candidates for `T`
    x.call_me();
    //~^ ERROR multiple applicable items in scope
}

fn main() {}
