//@ error-pattern: requires `copy` lang_item

#![feature(lang_items, no_core)]
#![no_core]
#![no_main]

#[lang = "pointeesized"]
pub trait PointeeSized {}

#[lang = "metasized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
trait Sized: MetaSized { }

struct S;

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    argc
}
