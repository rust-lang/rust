//! Regression test for issue <https://github.com/rust-lang/rust/issues/155241>.
//@ run-pass
//@ revisions: noopt opt
//@[noopt] compile-flags: -C opt-level=0
//@[opt] compile-flags: -C opt-level=3

#![feature(fn_traits, stmt_expr_attributes)]
#![expect(unused)]

#[derive(Copy, Clone)]
struct Thing {
    x: usize,
    y: usize,
    z: usize,
}

#[inline(never)]
fn opt_0() {
    let value = (Thing { x: 0, y: 0, z: 0 },);
    (|mut thing: Thing| {
        thing.z = 1;
    })
    .call(value);
    assert_eq!(value.0.z, 0);
}

#[inline(never)]
fn opt_3() {
    fn with(f: impl FnOnce(Vec<usize>)) {
        f(Vec::new())
    }
    with(|mut v| v.resize(2, 1));
    with(|v| {
        if v.len() != 0 {
            unreachable!();
        }
    });
}

#[inline(never)]
fn const_() {
    const VALUE: (Thing,) = (Thing { x: 0, y: 0, z: 0 },);

    (#[inline(never)]
    |mut thing: Thing| {
        thing.z = 1;
        std::hint::black_box(&mut thing.z);
        assert_eq!(thing.z, 1);
    })
    .call(VALUE);
}

fn main() {
    opt_0();
    opt_3();
    const_();
}
