// Like test `assoc-items.rs` but now utilizing unstable features.
// FIXME: Make use of (m)GCA assoc consts once they no longer ICE!
//@ compile-flags: -Zunstable-options --generate-link-to-definition
#![feature(return_type_notation)]

//@ has 'src/assoc_items_extra/assoc-items-extra.rs.html'

trait Trait0 {
    fn fn0() -> impl Sized;
    fn fn1() -> impl Sized;
}

fn item<T: Trait0>()
where
    //@ has - '//a[@href="#9"]' 'fn0'
    <T as Trait0>::fn0(..): Copy,   // Item, AssocFn,     Resolved
    // FIXME: Support this:
    //@ !has - '//a[@href="#10"]' 'fn1'
    T::fn1(..): Copy,               // Item, AssocFn,    TypeRelative
{}
