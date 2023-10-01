// only-x86_64
// check-fail

#![feature(lang_items, no_core, target_feature_11)]
#![no_core]

#[lang = "copy"]
pub trait Copy {}
#[lang = "sized"]
pub trait Sized {}

#[lang = "start"]
#[target_feature(enable = "avx2")]
//~^ ERROR `start` language item function is not allowed to have `#[target_feature]`
fn start<T>(_main: fn() -> T, _argc: isize, _argv: *const *const u8, _sigpipe: u8) -> isize {
    0
}

fn main() {}
