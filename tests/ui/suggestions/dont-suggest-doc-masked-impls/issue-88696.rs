// aux-build:masked-crate.rs
//
// This test case should ensure that miniz_oxide isn't
// suggested, since it's not a direct dependency.

#![feature(doc_masked)]

#[doc(masked)]
extern crate masked_crate;

fn a() -> Result<u64, i32> {
    Err(1)
}

fn b() -> Result<u32, i32> {
    a().into() //~ERROR [E0277]
}

fn main() {
    let _ = dbg!(b());
    // make sure crate actually gets loaded
    let _ = masked_crate::StreamResult;
}
