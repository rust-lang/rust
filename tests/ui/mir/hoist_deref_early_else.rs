//! Regression test for issue <https://github.com/rust-lang/rust/issues/159591>.
//! The null pointer `p` should never be dereferenced.
//@[noopt] run-pass
//@[opt] run-crash
//@ revisions: noopt opt
//@ check-run-results
//@[noopt] compile-flags: -C opt-level=0
//@[opt] compile-flags: -C opt-level=3

use std::hint::black_box;

#[inline(never)]
fn foo(q: u64, p: *const u64) -> u64 {
    unsafe {
        'a: {
            match q {
                1 => match *p {
                    1 => break 'a 100,
                    _ => {}
                },
                2 => match *p {
                    2 => break 'a 200,
                    _ => {}
                },
                _ => {}
            }
            999
        }
    }
}

fn main() {
    let q: u64 = black_box(3);
    let p: *const u64 = black_box(std::ptr::null());
    let r = foo(q, p);
    assert_eq!(999, black_box(r));
}
