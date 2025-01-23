//@ check-pass
//@ compile-flags: -Zexperimental-default-bounds

#![feature(auto_traits, lang_items, no_core, rustc_attrs, trait_alias)]
#![no_std]
#![no_core]

#[lang = "pointee_sized"]
trait PointeeSized {}

#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}

#[lang = "sized"]
trait Sized: MetaSized {}

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

fn main() {}
