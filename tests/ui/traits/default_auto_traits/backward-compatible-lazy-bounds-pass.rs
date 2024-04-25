//@ check-pass
//@ compile-flags: -Zexperimental-default-bounds

#![feature(auto_traits, lang_items, no_core, start, rustc_attrs, trait_alias)]
#![no_std]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[lang = "default_trait1"]
auto trait DefaultTrait1 {}

#[lang = "default_trait2"]
auto trait DefaultTrait2 {}

trait Trait<Rhs: ?Sized = Self> {}
trait Trait1 : Trait {}

trait Trait2 {
    type Type;
}
trait Trait3<T> = Trait2<Type = T>;

#[start]
fn main(_argc: isize, _argv: *const *const u8) -> isize { 0 }
