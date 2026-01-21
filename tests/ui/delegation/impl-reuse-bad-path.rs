#![allow(incomplete_features)]
#![feature(fn_delegation)]

mod unresolved {
    struct S;
    reuse impl unresolved for S { self.0 }
    //~^ ERROR failed to resolve: use of unresolved module or unlinked crate `unresolved`
    //~| ERROR cannot find trait `unresolved` in this scope

    trait T {}
    reuse impl T for unresolved { self.0 }
    //~^ ERROR empty glob delegation is not supported
    //~| ERROR cannot find type `unresolved` in this scope
}

mod wrong_entities {
    trait T {}
    struct Trait;
    struct S;

    reuse impl Trait for S { self.0 }
    //~^ ERROR expected trait, found struct `Trait`
    //~| ERROR expected trait, found struct `Trait`

    mod TraitModule {}
    reuse impl TraitModule for S { self.0 }
    //~^ ERROR expected trait, found module `TraitModule`
    //~| ERROR expected trait, found module `TraitModule`
}

fn main() {}
