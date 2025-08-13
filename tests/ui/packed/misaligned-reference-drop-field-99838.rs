// https://github.com/rust-lang/rust/issues/99838
//@ run-pass

use std::hint;

struct U16(#[allow(dead_code)] u16);

impl Drop for U16 {
    fn drop(&mut self) {
        // Prevent LLVM from optimizing away our alignment check.
        assert!(hint::black_box(self as *mut U16 as usize) % 2 == 0);
    }
}

struct HasDrop;

impl Drop for HasDrop {
    fn drop(&mut self) {}
}

struct Wrapper {
    _a: U16,
    b: HasDrop,
}

#[repr(packed)]
struct Misalign(#[allow(dead_code)] u8, Wrapper);

fn main() {
    let m = Misalign(
        0,
        Wrapper {
            _a: U16(10),
            b: HasDrop,
        },
    );
    // Put it somewhere definitely even (so the `a` field is definitely at an odd address).
    let m: ([u16; 0], Misalign) = ([], m);
    // Move out one field, so we run custom per-field drop logic below.
    let _x = m.1.1.b;
}
