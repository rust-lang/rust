//@ compile-flags: -Znext-solver
//@ ignore-test (see #114196)

trait Trait {
    type Gat<'lt>;
}
impl Trait for u8 {
    type Gat<'lt> = u8;
}

fn test<T: Trait, F: FnOnce(<T as Trait>::Gat<'_>) -> S + ?Sized, S>() {}

fn main() {
    // Proving `dyn FnOnce: FnOnce` requires making sure that all of the supertraits
    // of the trait and associated type bounds hold. We check this in
    // `predicates_for_object_candidate`, and eagerly replace projections using equality
    // which may generalize a type and emit a nested AliasRelate goal. Make sure that
    // we don't ICE in that case, and bubble that goal up to the caller.
    test::<u8, dyn FnOnce(<u8 as Trait>::Gat<'_>) + 'static, _>();
}
