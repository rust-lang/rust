// Check that we properly reconstruct sized bounds from the middle::ty IR to the surface syntax.

#![crate_name = "it"]
//@ aux-build: ext-sized-bounds.rs
extern crate ext_sized_bounds as ext;

// Ensure that we translate [<T as Sized>] to ``.
// Non-`Self` type params are implicitly `Sized`, so we can hide it.
//
//@ has it/fn.sized_param.html
//@ has - '//pre[@class="rust item-decl"]' "fn sized_param<T>()"
//@ !has - '//pre[@class="rust item-decl"]' "T: Sized"
pub use ext::sized_param;

// Ensure that we translate [<T as MetaSized>] to `T: ?Sized`.
// On stable, the user must've opted out of the implicit `Sized` bound using relaxed bound `?Sized`
// which will implicitly add the `MetaSized` bound (which is unstable in the surface language).
//
//@ has it/fn.relaxed_sized_on_param.html
//@ has - '//pre[@class="rust item-decl"]' "fn relaxed_sized_on_param<T>()where T: ?Sized"
pub use ext::relaxed_sized_on_param;

// Ensure that we don't drop the `T: Sized` bound on `func`. Previously, we didn't check if `T`
// actually belongs to the closest item and thought that it was an implicit bound which it isn't.
// issue: <https://github.com/rust-lang/rust/issues/144015>
//
//@ has it/trait.SizedOnParentParam.html
//@ has - '//*[@class="rust item-decl"]' \
//        "trait SizedOnParentParam<T>\
//        where \
//            T: ?Sized"
//@ has - '//*[@id="tymethod.func"]' \
//        "fn func()\
//        where \
//            T: Sized"
pub use ext::SizedOnParentParam;

// Ensure that we don't drop the `Self: Sized` bound on traits.
// Traits are *not* implicitly bounded by `Sized`. They're only implicitly bounded by `MetaSized`.
//
//@ has it/trait.SizedSelf.html
//@ has - '//*[@class="rust item-decl"]' 'trait SizedSelf: Sized {'
pub use ext::SizedSelf;

// Ensure that we don't drop the `Self: Sized` bound.
// First of all, `Self` type params of traits are not implicitly bounded by `Self`.
// Second of all, `Self` appears in a bound on an assoc item but `Self` belongs to the parent item,
// the trait meaning it's definitely user-written and not implicit.
// issue: <https://github.com/rust-lang/rust/issues/24183>
//
//@ has it/trait.SizedOnParentSelf.html
//@ has - '//*[@class="rust item-decl"]' "trait SizedOnParentSelf {"
//@ has - '//*[@id="method.func"]' \
// "fn func(self) -> Self\
// where \
//     Self: Sized"
pub use ext::SizedOnParentSelf;
