//@ check-pass
//@ compile-flags: -Znext-solver -Zassumptions-on-binders

#![crate_type = "lib"]

// When entering binders if we don't insert assumptions corresponding to the
// binder then we'll ICE when later trying to eagerly handle placeholders. This
// test checks that type relations involving higher ranked types insert assumptions
// for the binders of the higher ranked types.
//
// This is specifically checking that type relations used from *inside* the trait
// solver do this. In this case the type relation occurs as part of an `AliasRelate`
// involving the rigid alias: `<T as Trait>::Assoc<for<'b> fn('b ()) -> &'?0 ()>`.

trait Trait { type Assoc<T>; }

fn foo<T: Trait, U>(_: U) -> <T as Trait>::Assoc::<U> { loop {} }

fn mk<'a>() -> for<'b> fn(&'b ()) -> &'a () { loop {} }

fn bar<T: Trait>() {
    foo::<T, _>(mk());
}
