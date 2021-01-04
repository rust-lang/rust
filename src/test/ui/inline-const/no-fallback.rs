// test for #78132, see the comment in `fn typeck` for why this
// is supposed to fail.
#![feature(inline_const)]
#![allow(incomplete_features)]

fn main() {
    let _: usize = const { 0 };
    //~^ ERROR type annotations needed
}
