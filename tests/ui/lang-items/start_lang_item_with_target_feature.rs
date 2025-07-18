//@ only-x86_64
//@ check-fail

#![feature(lang_items, no_core)]
#![no_core]

#[lang = "copy"]
pub trait Copy {}

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[lang = "start"]
#[target_feature(enable = "avx2")]
//~^ ERROR `start` lang item function is not allowed to have `#[target_feature]`
fn start<T>(_main: fn() -> T, _argc: isize, _argv: *const *const u8, _sigpipe: u8) -> isize {
    0
}

fn main() {}
