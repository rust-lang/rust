// Regression test for issues #77763, #84579 and #102142.
#![crate_name = "main"]

//@ aux-build:assoc_item_trait_bounds.rs
//@ build-aux-docs
//@ ignore-cross-compile
extern crate assoc_item_trait_bounds as aux;

//@ has main/trait.Main.html
//@ has - '//*[@id="associatedtype.Out0"]' 'type Out0: Support<Item = ()>'
//@ has - '//*[@id="associatedtype.Out1"]' 'type Out1: Support<Item = Self::Item>'
//@ has - '//*[@id="associatedtype.Out2"]' 'type Out2<T>: Support<Item = T>'
//@ has - '//*[@id="associatedtype.Out3"]' 'type Out3: Support<Produce<()> = bool>'
//@ has - '//*[@id="associatedtype.Out4"]' 'type Out4<T>: Support<Produce<T> = T>'
//@ has - '//*[@id="associatedtype.Out5"]' "type Out5: Support<Output<'static> = &'static ()>"
//@ has - '//*[@id="associatedtype.Out6"]' "type Out6: for<'a> Support<Output<'a> = &'a ()>"
//@ has - '//*[@id="associatedtype.Out7"]' "type Out7: Support<Item = String, Produce<i32> = u32> + Unrelated"
//@ has - '//*[@id="associatedtype.Out8"]' "type Out8: Unrelated + Protocol<i16, Q1 = u128, Q0 = ()>"
//@ has - '//*[@id="associatedtype.Out9"]' "type Out9: FnMut(i32) -> bool + Clone"
//@ has - '//*[@id="associatedtype.Out10"]' "type Out10<'q>: Support<Output<'q> = ()>"
//@ has - '//*[@id="associatedtype.Out11"]' "type Out11: for<'r, 's> Helper<A<'s> = &'s (), B<'r> = ()>"
//@ has - '//*[@id="associatedtype.Out12"]' "type Out12: for<'w> Helper<B<'w> = Cow<'w, str>, A<'w> = bool>"
//@ has - '//*[@id="associatedtype.Out13"]' "type Out13: for<'fst, 'snd> Aid<'snd, Result<'fst> = &'fst mut str>"
//@ has - '//*[@id="associatedtype.Out14"]' "type Out14<P: Copy + Eq, Q: ?Sized>"
//@ has - '//*[@id="associatedtype.Out15"]' "type Out15: AsyncFnMut(i32) -> bool"
//
// Snapshots:
// Check that we don't render any where-clauses for the following associated types since
// all corresponding projection equality predicates should have already been re-sugared
// to associated type bindings:
//
//@ snapshot out0 - '//*[@id="associatedtype.Out0"]/*[@class="code-header"]'
//@ snapshot out2 - '//*[@id="associatedtype.Out2"]/*[@class="code-header"]'
//@ snapshot out9 - '//*[@id="associatedtype.Out9"]/*[@class="code-header"]'
//
//@ has - '//*[@id="tymethod.make"]' \
// "fn make<F>(_: F, _: impl FnMut(&str) -> bool)\
// where \
//     F: FnOnce(u32) -> String, \
//     Self::Out2<()>: Protocol<u8, Q0 = Self::Item, Q1 = ()>"
pub use aux::Main;

//@ has main/trait.Aid.html
//@ has - '//*[@id="associatedtype.Result"]' "type Result<'inter: 'src>"
pub use aux::Aid;

// Below, ensure that we correctly display generic parameters and where-clauses on
// associated types inside trait *impls*. More particularly, check that we don't render
// any bounds (here `Self::Alias<T>: ...`) as item bounds unlike all the trait test cases above.

//@ has main/struct.Implementor.html
//@ has - '//*[@id="associatedtype.Alias"]' \
// "type Alias<T: Eq> = T \
// where \
//     String: From<T>, \
//     <Implementor as Implementee>::Alias<T>: From<<Implementor as Implementee>::Alias<T>>"
pub use aux::Implementor;
