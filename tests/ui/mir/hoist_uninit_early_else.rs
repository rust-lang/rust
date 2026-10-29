//@ run-pass
//@ revisions: noopt opt disabled
//@[noopt] compile-flags: -C opt-level=0
//@[opt] compile-flags: -C opt-level=3
//@[disabled] compile-flags: -C opt-level=3 -Zmir-enable-passes=-EarlyOtherwiseBranch

use std::hint::black_box;

union U {
    a: u64,
    b: (),
}

#[inline(never)]
fn foo(q: u64) -> u64 {
    let mut u = U { b: () };
    match q {
        1 => u.a = 1,
        2 => u.a = 2,
        _ => {}
    }
    black_box(());
    let mut r = 999;
    // SAFETY: Each read of `u.a` is preceded by its initialization for the same value of `q`.
    unsafe {
        match q {
            1 => match u.a {
                1 => r = 100,
                _ => {}
            },
            2 => match u.a {
                2 => r = 200,
                _ => {}
            },
            _ => {}
        }
    }
    r
}

fn main() {
    for (q, expected) in [(0, 999), (1, 100), (2, 200), (3, 999), (u64::MAX, 999)] {
        assert_eq!(foo(black_box(q)), expected);
    }
}
