// regression test for #137813 where we would assume all constants in the type system
// cannot contain inference variables, even though associated const equality syntax
// was still lowered without the feature gate enabled.

trait AssocConst {
    const A: u8;
}

impl<T> AssocConst for (T,) {
    const A: u8 = 0;
}

trait Trait {}

impl<U> Trait for () where (U,): AssocConst<A = { 0 }> {}
//~^ ERROR associated const equality is incomplete
//~| ERROR the type parameter `U` is not constrained by the impl trait

fn foo()
where
    (): Trait,
    //~^ ERROR type mismatch resolving
{
}

fn main() {}
