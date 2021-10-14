// run-pass
// only-x86_64

#![feature(const_eval_select)]
use std::intrinsics::const_eval_select;
use std::arch::x86_64::*;
use std::mem::transmute;

const fn eq_ct(x: [i32; 4], y: [i32; 4]) -> bool {
    x[0] == y[0] && x[1] == y[1] && x[2] == y[2] && x[3] == y[3]
}

fn eq_rt(x: [i32; 4], y: [i32; 4]) -> bool {
    unsafe {
        let x = _mm_loadu_si128(&x as *const _ as *const _);
        let y = _mm_loadu_si128(&y as *const _ as *const _);
        let r = _mm_cmpeq_epi32(x, y);
        let r = _mm_movemask_ps(transmute(r) );
        r == 0b1111
    }
}

const fn eq(x: [i32; 4], y: [i32; 4]) -> bool {
    unsafe {
        const_eval_select((x, y), eq_ct, eq_rt)
    }
}

fn main() {
    const X: bool = eq([0, 1, 2, 3], [0, 1, 2, 3]);
    assert_eq!(X, true);
    let x = eq([0, 1, 2, 3], [0, 1, 2, 3]);
    assert_eq!(x, true);

    const Y: bool = eq([0, 1, 2, 3], [0, 1, 3, 2]);
    assert_eq!(Y, false);
    let y = eq([0, 1, 2, 3], [0, 1, 3, 2]);
    assert_eq!(y, false);
}
