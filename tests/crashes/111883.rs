//@ known-bug: #111883
#![crate_type = "lib"]
#![feature(arbitrary_self_types, no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}
#[lang = "receiver"]
trait Receiver {}
#[lang = "dispatch_from_dyn"]
trait DispatchFromDyn<T> {}
impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<&'a U> for &'a T {}
#[lang = "unsize"]
trait Unsize<T: ?Sized> {}
#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T: ?Sized> {}
impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}

#[lang = "drop_in_place"]
fn drop_in_place_fn<T>(a: &dyn Trait2<T>) {}

pub trait Trait1 {
    fn foo(&self);
}

pub struct Type1;

impl Trait1 for Type1 {
    fn foo(&self) {}
}

pub trait Trait2<T> {}

pub fn bar1() {
    let a = Type1;
    let b = &a as &dyn Trait1;
    b.foo();
}
