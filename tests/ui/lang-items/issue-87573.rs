// Regression test for #87573, ensures that duplicate lang items or invalid generics
// for lang items doesn't cause ICE.

#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "lib"]

pub static STATIC_BOOL: bool = true;

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "size_of_val"]
pub trait SizeOfVal: PointeeSized {}

#[lang = "sized"]
trait Sized: SizeOfVal {}

#[lang = "copy"]
trait Copy {}

#[lang = "sync"]
trait Sync {}
impl Sync for bool {}

#[lang = "drop_in_place"]
//~^ ERROR: `drop_in_place` lang item must be applied to a function with at least 1 generic argument
fn drop_fn() {
    while false {}
}

#[lang = "start"]
//~^ ERROR: `start` lang item must be applied to a function with 1 generic argument
fn start(){}
