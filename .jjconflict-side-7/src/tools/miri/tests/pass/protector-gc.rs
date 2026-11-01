// When we pop a stack frame with weak protectors, we need to check if the protected pointer's
// allocation is still live. If the provenance GC only knows about the BorTag that is protected,
// we can ICE. This test checks that we don't.
// See https://github.com/rust-lang/miri/issues/3228

#[path = "../utils/mod.rs"]
mod utils;

#[allow(unused)]
fn oof(mut b: Box<u8>) {
    b = Box::new(0u8);
    utils::run_provenance_gc();
}

fn main() {
    oof(Box::new(0u8));
}
