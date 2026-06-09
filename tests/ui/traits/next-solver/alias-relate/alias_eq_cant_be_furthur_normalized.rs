//@ check-pass
//@ compile-flags: -Znext-solver

// check that a goal such as `alias-eq(<T as TraitB>::Assoc<bool>, <T as TraitB>::Assoc<?0>)`
// succeeds with a constraint that `?0 = bool`

// FIXME(deferred_projection_equality): add a test that this is true during coherence

trait TraitA {}

trait TraitB {
    type Assoc<T: ?Sized>;
}

impl<T: TraitB> TraitA for (T, T::Assoc<bool>) {}

impl TraitB for i32 {
    type Assoc<T: ?Sized> = u32;
}

fn needs_a<T: TraitA>() {}

fn bar<T: TraitB>() {
    needs_a::<(T, <T as TraitB>::Assoc<_>)>();
}

fn main() {
    bar::<i32>();
}
