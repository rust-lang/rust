#![feature(lang_items, no_core, const_trait_impl)]
#![no_core]
#![no_main]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
#[const_trait]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
#[const_trait]
trait Sized: MetaSized { }

struct S;

#[no_mangle]
extern "C" fn main(argc: i32, _argv: *const *const u8) -> i32 {
    argc //~ ERROR requires `copy` lang_item
}
