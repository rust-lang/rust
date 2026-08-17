#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

// This tests verifies that Enzyme can differentiate the iterator and window version of the for
// loops given below. Iterators (especially the windows use here) cause a lot of extra abstractions
// and indirections. Without extra typetree hints, Enzyme failed to differentiate them in debug
// mode.

//@revisions: tt no_tt
//@[tt] compile-flags: -Z autodiff=Enable
//@[no_tt] compile-flags: -Z autodiff=Enable,NoTT
//@[no_tt] build-fail

#[unsafe(no_mangle)]
#[inline(never)]
#[autodiff_reverse(f_rev, 2, Duplicated, Const, Duplicated)]
fn f(x: &[f64; 3], args: &[f64; 3], y: &mut [f64; 2]) {
    y[0] = x.iter().map(|i| args[0] * i.powi(2)).sum();
    y[1] = x
        .windows(2)
        .map(|w| (args[1] - w[0]).powi(2) + args[2] * (w[1] - w[0].powi(2)).powi(2))
        .sum();
    // The iterators above are equivalent to the two following for loops.
    // for i in 0..3 {
    //     y[0] += args[0] * x[i].powi(2);
    // }
    // for i in 0..2 {
    //     y[1] += (args[1] - x[i]).powi(2) + args[2] * (x[i + 1] - x[i].powi(2)).powi(2);
    // }
}

// Not generally recommended, but since we rewrite llvm-ir, it should be good enough.
fn assert_abs_diff_eq<const N: usize>(x: &[f64; N], y: &[f64; N]) {
    for i in 0..N {
        assert_eq!(x[i], y[i]);
    }
}

fn main() {
    let x = [3.0, 5.0, 7.0];
    let args = [2.0, 1.0, 100.0];

    let mut vjp = ([0.0; 3], [0.0; 3]);
    let mut y = [0.0; 2];
    let mut dy = ([1.0, 0.0], [0.0, 1.0]);

    f_rev(&x, &mut vjp.0, &mut vjp.1, &args, &mut y, &mut dy.0, &mut dy.1);

    assert_abs_diff_eq::<2>(&y, &[166.0, 34020.0]);
    assert_abs_diff_eq::<3>(&vjp.0, &[12.0, 20.0, 28.0]);
    assert_abs_diff_eq::<3>(&vjp.1, &[4804.0, 35208.0, -3600.0]);
}
