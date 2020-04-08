//! Tests that generators that cannot return or unwind don't have unnecessary
//! panic branches.

// compile-flags: -Zno-landing-pads

#![feature(generators, generator_trait)]

struct HasDrop;

impl Drop for HasDrop {
    fn drop(&mut self) {}
}

fn callee() {}

// EMIT_MIR rustc.main-{{closure}}.generator_resume.0.mir
fn main() {
    let _gen = |_x: u8| {
        let _d = HasDrop;
        loop {
            yield;
            callee();
        }
    };
}
