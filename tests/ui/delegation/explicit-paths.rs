#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait {
    fn foo1(&self, x: i32) -> i32 { x }
    fn foo2(x: i32) -> i32 { x }
}

struct F;
impl Trait for F {}
struct S(F);

pub mod to_reuse {
    pub fn foo3() {}
}

impl F {
    fn foo4(&self) {}
}

mod fn_to_other {
    use super::*;

    reuse Trait::foo1;
    reuse <S as Trait>::foo2;
    reuse to_reuse::foo3;
    reuse S::foo4;
    //~^ ERROR cannot find function `foo4` in `S`
}

mod inherent_impl_assoc_fn_to_other {
    use crate::*;

    impl S {
        reuse Trait::foo1 { self.0 }
        reuse <S as Trait>::foo2;
        reuse to_reuse::foo3;
        reuse F::foo4 { &self.0 }
        //~^ ERROR cannot find function `foo4` in `F`
    }
}

mod trait_impl_assoc_fn_to_other {
    use crate::*;

    impl Trait for S {
        reuse Trait::foo1 { self.0 }
        reuse <F as Trait>::foo2;
        reuse to_reuse::foo3;
        //~^ ERROR method `foo3` is not a member of trait `Trait`
        reuse F::foo4 { &self.0 }
        //~^ ERROR method `foo4` is not a member of trait `Trait`
        //~| ERROR cannot find function `foo4` in `F`
    }
}

mod trait_assoc_fn_to_other {
    use crate::*;

    trait Trait2 : Trait {
        reuse <F as Trait>::foo1 { self }
        //~^ ERROR mismatched types
        reuse <F as Trait>::foo2;
        reuse to_reuse::foo3;
        reuse F::foo4 { &F }
        //~^ ERROR cannot find function `foo4` in `F`
    }
}

mod type_mismatch {
    use crate::*;

    struct S2;
    impl Trait for S {
    //~^ ERROR conflicting implementations of trait `Trait` for type `S`
        reuse <S2 as Trait>::foo1;
        //~^ ERROR mismatched types
        //~| ERROR the trait bound `S2: Trait` is not satisfied
    }
}

fn main() {}
