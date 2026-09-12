//@ run-pass

// Bad regression test for https://github.com/rust-lang/rust/issues/159035
// This should probably be replaced with a codegen test, though that might need to be in LLVM,
// as it seems that the miscompilation may occur on correct IR misoptimized by SimplifyCFG.

#![allow(dead_code)]

enum Inner {
    A(u32),
    B(u32),
}
struct Big {
    _pad: u64,
    inner: Inner,
}
struct Small {
    a: u16,
    b: u16,
    _f: fn(),
}
enum Checksum {
    X(Big),
    Y(Small),
}
impl Checksum {
    fn finalize(self) -> u32 {
        match self {
            Checksum::X(h) => match h.inner {
                Inner::A(s) => s,
                Inner::B(s) => s,
            },
            Checksum::Y(s) => (u32::from(s.b) << 16) | u32::from(s.a),
        }
    }
}
#[inline(never)]
fn run(c: Option<Checksum>) -> Option<u32> {
    c.map(|c| c.finalize())
}
fn main() {
    println!("{:?}", run(std::hint::black_box(None)));
}
