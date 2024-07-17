//@ known-bug: #107975
//@ compile-flags: -Copt-level=2
//@ run-pass
//@ check-run-results
//@ normalize-stdout-test: "\d+" -> "<..>"

// Based on https://github.com/rust-lang/rust/issues/107975#issuecomment-1432161340

#[inline(never)]
fn cmp(a: usize, b: usize) -> bool {
    a == b
}

#[inline(always)]
fn cmp_in(a: usize, b: usize) -> bool {
    a == b
}

fn main() {
    let a = {
        let v = 0;
        &v as *const _ as usize
    };
    let b = {
        let v = 0;
        &v as *const _ as usize
    };
    assert_eq!(a.to_string(), b.to_string());
    println!("{a:?} == {b:?} -> ==: {}, cmp_in: {}, cmp: {}", a == b, cmp_in(a, b), cmp(a, b));
}
