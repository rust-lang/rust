// Regression test for issues #77763, #84579 and #102142.
#![crate_name = "main"]

// aux-build:assoc_item_trait_bounds_with_bindings.rs
// build-aux-docs
// ignore-cross-compile
extern crate assoc_item_trait_bounds_with_bindings as aux;

// FIXME(fmease): Don't render an incorrect `T: ?Sized` where-clause for parameters
//                of GATs like `Main::Out{2,4}`. Add a snapshot test once it's fixed.
// FIXME(fmease): Print the `for<>` parameter list in the bounds of
//                `Main::Out{6,11,12}`.

// @has main/trait.Main.html
// @has - '//*[@id="associatedtype.Out0"]' 'type Out0: Support<Item = ()>'
// @has - '//*[@id="associatedtype.Out1"]' 'type Out1: Support<Item = Self::Item>'
// @has - '//*[@id="associatedtype.Out2"]' 'type Out2<T>: Support<Item = T>'
// @has - '//*[@id="associatedtype.Out3"]' 'type Out3: Support<Produce<()> = bool>'
// @has - '//*[@id="associatedtype.Out4"]' 'type Out4<T>: Support<Produce<T> = T>'
// @has - '//*[@id="associatedtype.Out5"]' "type Out5: Support<Output<'static> = &'static ()>"
// @has - '//*[@id="associatedtype.Out6"]' "type Out6: Support<Output<'a> = &'a ()>"
// @has - '//*[@id="associatedtype.Out7"]' "type Out7: Support<Item = String, Produce<i32> = u32> + Unrelated"
// @has - '//*[@id="associatedtype.Out8"]' "type Out8: Unrelated + Protocol<i16, Q1 = u128, Q0 = ()>"
// @has - '//*[@id="associatedtype.Out9"]' "type Out9: FnMut(i32) -> bool + Clone"
// @has - '//*[@id="associatedtype.Out10"]' "type Out10<'q>: Support<Output<'q> = ()>"
// @has - '//*[@id="associatedtype.Out11"]' "type Out11: Helper<A<'s> = &'s (), B<'r> = ()>"
// @has - '//*[@id="associatedtype.Out12"]' "type Out12: Helper<B<'w> = Cow<'w, str>, A<'w> = bool>"
//
// Snapshots: Check that we do not render any where-clauses for those associated types since all of
// the trait bounds contained within were moved to the bounds of the respective item.
//
// @snapshot out0 - '//*[@id="associatedtype.Out0"]/*[@class="code-header"]'
// @snapshot out9 - '//*[@id="associatedtype.Out9"]/*[@class="code-header"]'
//
// @has - '//*[@id="tymethod.make"]' \
// "fn make<F>(F, impl FnMut(&str) -> bool)\
// where \
//     F: FnOnce(u32) -> String, \
//     Self::Out2<()>: Protocol<u8, Q0 = Self::Item, Q1 = ()>"
pub use aux::Main;
