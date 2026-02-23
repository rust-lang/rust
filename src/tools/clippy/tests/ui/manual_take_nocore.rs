//@ check-pass
#![feature(no_core, lang_items)]
#![no_core]
#![allow(clippy::missing_safety_doc)]
#![warn(clippy::manual_take)]

#[link(name = "c")]
unsafe extern "C" {}

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}
#[lang = "copy"]
pub trait Copy {}
#[lang = "freeze"]
pub unsafe trait Freeze {}

#[lang = "start"]
fn start<T>(_main: fn() -> T, _argc: isize, _argv: *const *const u8, _sigpipe: u8) -> isize {
    0
}

fn main() {
    let mut x = true;
    // this should not lint because we don't have std nor core
    let _manual_take = if x {
        x = false;
        true
    } else {
        false
    };
}
