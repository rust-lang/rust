// revisions: normal over_aligned
//[normal]failure-status:101
// normalize-stderr-test "\n\nnote: .*" -> ""
// normalize-stderr-test "thread 'rustc' .*" -> ""
// normalize-stderr-test " +[0-9]+:.*\n" -> ""
// normalize-stderr-test " +at .*\n" -> ""

#![feature(dyn_star)]
//~^ WARN the feature `dyn_star` is incomplete and may not be safe to use and/or cause compiler crashes

use std::fmt::Debug;

#[cfg_attr(over_aligned, repr(C, align(1024)))]
#[cfg_attr(not(over_aligned), repr(C))]
#[derive(Debug)]
struct AlignedUsize(usize);

#[rustfmt::skip]
fn main() {
    let x = AlignedUsize(12) as dyn *Debug;
    //[over_aligned]~^ ERROR `AlignedUsize` needs to have the same alignment and size as a pointer
    //[normal]~^^ ERROR primitive read not possible for type: AlignedUsize
}
