// only-x86_64

#![feature(start)]
#![feature(target_feature_11)]

#[start]
#[target_feature(enable = "avx2")]
//~^ ERROR `start` is not allowed to have `#[target_feature]`
fn start(_argc: isize, _argv: *const *const u8) -> isize { 0 }
