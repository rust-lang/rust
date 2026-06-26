//@ check-pass
//@ compile-flags: -Znext-solver

// A regression test for trait-system-refactor-initiative#184.
//
// When adding nested goals we replace aliases with infer vars
// and add `AliasRelate` goals to constrain them. When doing this
// for `NormalizesTo` goals, we then first tries to prove the
// `NormalizesTo` goal and then normalized the nested aliases.

trait Trait<T> {
    type Assoc;
}
impl<T, U> Trait<U> for T {
    type Assoc = ();
}

trait Id {
    type This;
}
impl<T> Id for T {
    type This = T;
}
trait Relate<T> {
    type Alias;
}
impl<T, U> Relate<U> for T {
    type Alias = <T as Trait<<U as Id>::This>>::Assoc;
}


fn guide_me<T: Trait<u32>>() {
    // Normalizing `<T as Relate<i32>>::Alias` relates the associated type with an unconstrained
    // term. This resulted in a `NormalizesTo(<T as Trait<<U as Id>::This>>::Assoc, ?x)` goal.
    // We replace `<i32 as Id>::This` with an infer var `?y`, resulting in the following goals:
    // - `NormalizesTo(<T as Trait<?y>::Assoc, ?x)`
    // - `AliasRelate(<i32 as Id>::This, ?y)`
    //
    // When proving the `NormalizesTo` goal first, we incompletely constrain `?y` to `u32`,
    // causing an unexpected type mismatch.
    let _: <T as Relate<i32>>::Alias;
}

fn main() {}
