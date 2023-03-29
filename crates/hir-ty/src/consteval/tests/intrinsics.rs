use super::*;

#[test]
fn size_of() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn size_of<T>() -> usize;
        }

        const GOAL: usize = size_of::<i32>();
        "#,
        4,
    );
}

#[test]
fn transmute() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn transmute<T, U>(e: T) -> U;
        }

        const GOAL: i32 = transmute((1i16, 1i16));
        "#,
        0x00010001,
    );
}

#[test]
fn const_eval_select() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn const_eval_select<ARG, F, G, RET>(arg: ARG, called_in_const: F, called_at_rt: G) -> RET
            where
                G: FnOnce<ARG, Output = RET>,
                F: FnOnce<ARG, Output = RET>;
        }

        const fn in_const(x: i32, y: i32) -> i32 {
            x + y
        }

        fn in_rt(x: i32, y: i32) -> i32 {
            x + y
        }

        const GOAL: i32 = const_eval_select((2, 3), in_const, in_rt);
        "#,
        5,
    );
}

#[test]
fn wrapping_add() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn wrapping_add<T>(a: T, b: T) -> T;
        }

        const GOAL: u8 = wrapping_add(10, 250);
        "#,
        4,
    );
}

#[test]
fn offset() {
    check_number(
        r#"
        //- minicore: coerce_unsized, index, slice
        extern "rust-intrinsic" {
            pub fn offset<T>(dst: *const T, offset: isize) -> *const T;
        }

        const GOAL: u8 = unsafe {
            let ar: &[(u8, u8, u8)] = &[
                (10, 11, 12),
                (20, 21, 22),
                (30, 31, 32),
                (40, 41, 42),
                (50, 51, 52),
            ];
            let ar: *const [(u8, u8, u8)] = ar;
            let ar = ar as *const (u8, u8, u8);
            let element = *offset(ar, 2);
            element.1
        };
        "#,
        31,
    );
}

#[test]
fn arith_offset() {
    check_number(
        r#"
        //- minicore: coerce_unsized, index, slice
        extern "rust-intrinsic" {
            pub fn arith_offset<T>(dst: *const T, offset: isize) -> *const T;
        }

        const GOAL: u8 = unsafe {
            let ar: &[(u8, u8, u8)] = &[
                (10, 11, 12),
                (20, 21, 22),
                (30, 31, 32),
                (40, 41, 42),
                (50, 51, 52),
            ];
            let ar: *const [(u8, u8, u8)] = ar;
            let ar = ar as *const (u8, u8, u8);
            let element = *arith_offset(arith_offset(ar, 102), -100);
            element.1
        };
        "#,
        31,
    );
}

#[test]
fn copy_nonoverlapping() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize);
        }

        const GOAL: u8 = unsafe {
            let mut x = 2;
            let y = 5;
            copy_nonoverlapping(&y, &mut x, 1);
            x
        };
        "#,
        5,
    );
}

#[test]
fn copy() {
    check_number(
        r#"
        //- minicore: coerce_unsized, index, slice
        extern "rust-intrinsic" {
            pub fn copy<T>(src: *const T, dst: *mut T, count: usize);
        }

        const GOAL: i32 = unsafe {
            let mut x = [1i32, 2, 3, 4, 5];
            let y = (&mut x as *mut _) as *mut i32;
            let z = (y as usize + 4) as *const i32;
            copy(z, y, 4);
            x[0] + x[1] + x[2] + x[3] + x[4]
        };
        "#,
        19,
    );
}
