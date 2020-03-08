//! Tests that generators that cannot return or unwind don't have unnecessary
//! panic branches.

// compile-flags: -Zno-landing-pads

#![feature(generators, generator_trait)]

struct HasDrop;

impl Drop for HasDrop {
    fn drop(&mut self) {}
}

fn callee() {}

fn main() {
    let _gen = |_x: u8| {
        let _d = HasDrop;
        loop {
            yield;
            callee();
        }
    };
}

// END RUST SOURCE

// START rustc.main-{{closure}}.generator_resume.0.mir
// bb0: {
//     ...
//     switchInt(move _11) -> [0u32: bb1, 3u32: bb5, otherwise: bb6];
// }
// ...
// END rustc.main-{{closure}}.generator_resume.0.mir
