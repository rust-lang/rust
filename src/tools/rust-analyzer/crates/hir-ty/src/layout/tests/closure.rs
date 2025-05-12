#![allow(clippy::match_single_binding)]
#![allow(clippy::no_effect)]

use crate::size_and_align_expr;

#[test]
fn zero_capture_simple() {
    size_and_align_expr! {
        |x: i32| x + 2
    }
}

#[test]
fn move_simple() {
    size_and_align_expr! {
        minicore: copy;
        stmts: []
        let y: i32 = 5;
        move |x: i32| {
            x + y
        }
    }
}

#[test]
fn ref_simple() {
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            let y: i32 = 5;
        ]
        |x: i32| {
            x + y
        }
    }
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            let mut y: i32 = 5;
        ]
        |x: i32| {
            y += x;
            y
        }
    }
    size_and_align_expr! {
        minicore: copy, deref_mut;
        stmts: [
            let y: &mut i32 = &mut 5;
        ]
        |x: i32| {
            *y += x;
        }
    }
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            struct X(i32, i64);
            let x: X = X(2, 6);
        ]
        || {
            x
        }
    }
    size_and_align_expr! {
        minicore: copy, deref_mut;
        stmts: [
            struct X(i32, i64);
            let x: &mut X = &mut X(2, 6);
        ]
        || {
            x.0 as i64 + x.1
        }
    }
}

#[test]
fn ref_then_mut_then_move() {
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            struct X(i32, i64);
            let mut x: X = X(2, 6);
        ]
        || {
            &x;
            &mut x;
            x;
        }
    }
}

#[test]
fn nested_closures() {
    size_and_align_expr! {
        || {
            || {
                || {
                    let x = 2;
                    move || {
                        move || {
                            x
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn capture_specific_fields2() {
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            let x = &mut 2;
        ]
        || {
            *x = 5;
            &x;
        }
    }
}

#[test]
fn capture_specific_fields() {
    size_and_align_expr! {
        struct X(i64, i32, (u8, i128));
        let y: X = X(2, 5, (7, 3));
        move |x: i64| {
            y.0 + x + (y.2 .0 as i64)
        }
    }
    size_and_align_expr! {
        struct X(i64, i32, (u8, i128));
        let y: X = X(2, 5, (7, 3));
        move |x: i64| {
            let _ = &y;
            y.0 + x + (y.2 .0 as i64)
        }
    }
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            struct X(i64, i32, (u8, i128));
            let y: X = X(2, 5, (7, 3));
        ]
        let y = &y;
        move |x: i64| {
            y.0 + x + (y.2 .0 as i64)
        }
    }
    size_and_align_expr! {
        struct X(i64, i32, (u8, i128));
        let y: X = X(2, 5, (7, 3));
        move |x: i64| {
            let X(a, _, (b, _)) = y;
            a + x + (b as i64)
        }
    }
    size_and_align_expr! {
        struct X(i64, i32, (u8, i128));
        let y = &&X(2, 5, (7, 3));
        move |x: i64| {
            let X(a, _, (b, _)) = y;
            *a + x + (*b as i64)
        }
    }
    size_and_align_expr! {
        struct X(i64, i32, (u8, i128));
        let y: X = X(2, 5, (7, 3));
        move |x: i64| {
            match y {
                X(a, _, (b, _)) => a + x + (b as i64),
            }
        }
    }
    size_and_align_expr! {
        struct X(i64, i32, (u8, i128));
        let y: X = X(2, 5, (7, 3));
        move |x: i64| {
            let X(a @ 2, _, (b, _)) = y else { return 5 };
            a + x + (b as i64)
        }
    }
}

#[test]
fn match_pattern() {
    size_and_align_expr! {
        struct X(i64, i32, (u8, i128));
        let _y: X = X(2, 5, (7, 3));
        move |x: i64| {
            x
        }
    }
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            struct X(i64, i32, (u8, i128));
            let y: X = X(2, 5, (7, 3));
        ]
        |x: i64| {
            match y {
                X(_a, _, _c) => x,
            }
        }
    }
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            struct X(i64, i32, (u8, i128));
            let y: X = X(2, 5, (7, 3));
        ]
        |x: i64| {
            match y {
                _y => x,
            }
        }
    }
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            struct X(i64, i32, (u8, i128));
            let y: X = X(2, 5, (7, 3));
        ]
        |x: i64| {
            match y {
                ref _y => x,
            }
        }
    }
}

#[test]
fn ellipsis_pattern() {
    size_and_align_expr! {
        struct X(i8, u16, i32, u64, i128, u8);
        let y: X = X(1, 2, 3, 4, 5, 6);
        move |_: i64| {
            let X(_a, .., _b, _c) = y;
        }
    }
    size_and_align_expr! {
        struct X { a: i32, b: u8, c: i128}
        let y: X = X { a: 1, b: 2, c: 3 };
        move |_: i64| {
            let X { a, b, .. } = y;
            _ = (a, b);
        }
    }
    size_and_align_expr! {
        let y: (&&&(i8, u16, i32, u64, i128, u8), u16, i32, u64, i128, u8) = (&&&(1, 2, 3, 4, 5, 6), 2, 3, 4, 5, 6);
        move |_: i64| {
            let ((_a, .., _b, _c), .., _e, _f) = y;
        }
    }
}

#[test]
fn regression_15623() {
    size_and_align_expr! {
        let a = 2;
        let b = 3;
        let c = 5;
        move || {
            let 0 = a else { return b; };

            c
        }
    }
}
