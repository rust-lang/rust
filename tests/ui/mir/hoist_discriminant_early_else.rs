// Regression test for #95162: the fallback path must not inspect q's invalid discriminant.
//@ run-pass
//@ revisions: noopt opt
//@[noopt] compile-flags: -C opt-level=0
//@[opt] compile-flags: -C opt-level=3

use std::hint::black_box;

#[repr(u8)]
enum E {
    A = 1,
    B = 2,
}

#[inline(never)]
fn foo(p: u8) -> u64 {
    let mut q = if p == 2 { E::B } else { E::A };
    if p != 1 && p != 2 {
        // SAFETY: E occupies one byte. q is only inspected on paths that skip this write.
        unsafe { (&raw mut q).cast::<u8>().write(10) };
    }
    black_box(());
    match p {
        1 => match q {
            E::A => 100,
            _ => 999,
        },
        2 => match q {
            E::B => 200,
            _ => 999,
        },
        _ => 999,
    }
}

fn main() {
    for (p, expected) in [(0, 999), (1, 100), (2, 200), (3, 999), (u8::MAX, 999)] {
        assert_eq!(foo(black_box(p)), expected);
    }
}
