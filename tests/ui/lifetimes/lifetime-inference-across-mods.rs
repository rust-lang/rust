//! regression test for <https://github.com/rust-lang/rust/issues/11529>
//@ run-pass
//@ aux-build:lifetime-inference-across-mods.rs


extern crate lifetime_inference_across_mods as a;

fn main() {
    let one = 1;
    let _a = a::A(&one);
}
