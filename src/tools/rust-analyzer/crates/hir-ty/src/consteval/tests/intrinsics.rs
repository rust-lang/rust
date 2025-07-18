use super::*;

#[test]
fn size_of() {
    check_number(
        r#"
        #[rustc_intrinsic]
        pub fn size_of<T>() -> usize;

        const GOAL: usize = size_of::<i32>();
        "#,
        4,
    );
}

#[test]
fn size_of_val() {
    check_number(
        r#"
        //- minicore: coerce_unsized
        #[rustc_intrinsic]
        pub fn size_of_val<T: ?Sized>(_: *const T) -> usize;

        struct X(i32, u8);

        const GOAL: usize = size_of_val(&X(1, 2));
        "#,
        8,
    );
    check_number(
        r#"
        //- minicore: coerce_unsized
        #[rustc_intrinsic]
        pub fn size_of_val<T: ?Sized>(_: *const T) -> usize;

        const GOAL: usize = {
            let it: &[i32] = &[1, 2, 3];
            size_of_val(it)
        };
        "#,
        12,
    );
    check_number(
        r#"
        //- minicore: coerce_unsized, transmute
        use core::mem::transmute;

        #[rustc_intrinsic]
        pub fn size_of_val<T: ?Sized>(_: *const T) -> usize;

        struct X {
            x: i64,
            y: u8,
            t: [i32],
        }

        const GOAL: usize = unsafe {
            let y: &X = transmute([0usize, 3]);
            size_of_val(y)
        };
        "#,
        24,
    );
    check_number(
        r#"
        //- minicore: coerce_unsized, transmute
        use core::mem::transmute;

        #[rustc_intrinsic]
        pub fn size_of_val<T: ?Sized>(_: *const T) -> usize;

        struct X {
            x: i32,
            y: i64,
            t: [u8],
        }

        const GOAL: usize = unsafe {
            let y: &X = transmute([0usize, 15]);
            size_of_val(y)
        };
    "#,
        32,
    );
    check_number(
        r#"
        //- minicore: coerce_unsized, fmt, builtin_impls, dispatch_from_dyn
        #[rustc_intrinsic]
        pub fn size_of_val<T: ?Sized>(_: *const T) -> usize;

        const GOAL: usize = {
            let x: &i16 = &5;
            let y: &dyn core::fmt::Debug = x;
            let z: &dyn core::fmt::Debug = &y;
            size_of_val(x) + size_of_val(y) * 10 + size_of_val(z) * 100
        };
        "#,
        1622,
    );
    check_number(
        r#"
        //- minicore: coerce_unsized
        #[rustc_intrinsic]
        pub fn size_of_val<T: ?Sized>(_: *const T) -> usize;

        const GOAL: usize = {
            size_of_val("salam")
        };
        "#,
        5,
    );
}

#[test]
fn align_of_val() {
    check_number(
        r#"
        //- minicore: coerce_unsized
        #[rustc_intrinsic]
        pub fn align_of_val<T: ?Sized>(_: *const T) -> usize;

        struct X(i32, u8);

        const GOAL: usize = align_of_val(&X(1, 2));
        "#,
        4,
    );
    check_number(
        r#"
        //- minicore: coerce_unsized
        #[rustc_intrinsic]
        pub fn align_of_val<T: ?Sized>(_: *const T) -> usize;

        const GOAL: usize = {
            let x: &[i32] = &[1, 2, 3];
            align_of_val(x)
        };
        "#,
        4,
    );
}

#[test]
fn type_name() {
    check_str(
        r#"
        #[rustc_intrinsic]
        pub fn type_name<T: ?Sized>() -> &'static str;

        const GOAL: &str = type_name::<i32>();
        "#,
        "i32",
    );
    check_str(
        r#"
        #[rustc_intrinsic]
        pub fn type_name<T: ?Sized>() -> &'static str;

        mod mod1 {
            pub mod mod2 {
                pub struct Ty;
            }
        }

        const GOAL: &str = type_name::<mod1::mod2::Ty>();
        "#,
        "mod1::mod2::Ty",
    );
}

#[test]
fn transmute() {
    check_number(
        r#"
        #[rustc_intrinsic]
        pub fn transmute<T, U>(e: T) -> U;

        const GOAL: i32 = transmute((1i16, 1i16));
        "#,
        0x00010001,
    );
}

#[test]
fn read_via_copy() {
    check_number(
        r#"
        #[rustc_intrinsic]
        pub fn read_via_copy<T>(e: *const T) -> T;
        #[rustc_intrinsic]
        pub fn volatile_load<T>(e: *const T) -> T;

        const GOAL: i32 = {
            let x = 2;
            read_via_copy(&x) + volatile_load(&x)
        };
        "#,
        4,
    );
}

#[test]
fn const_eval_select() {
    check_number(
        r#"
        //- minicore: fn
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
fn ptr_offset_from() {
    check_number(
        r#"
        //- minicore: index, slice, coerce_unsized
        extern "rust-intrinsic" {
            pub fn ptr_offset_from<T>(ptr: *const T, base: *const T) -> isize;
            pub fn ptr_offset_from_unsigned<T>(ptr: *const T, base: *const T) -> usize;
        }

        const GOAL: isize = {
            let x = [1, 2, 3, 4, 5i32];
            let r1 = -ptr_offset_from(&x[0], &x[4]);
            let r2 = ptr_offset_from(&x[3], &x[1]);
            let r3 = ptr_offset_from_unsigned(&x[3], &x[0]) as isize;
            r3 * 100 + r2 * 10 + r1
        };
        "#,
        324,
    );
}

#[test]
fn saturating() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn saturating_add<T>(a: T, b: T) -> T;
        }

        const GOAL: u8 = saturating_add(10, 250);
        "#,
        255,
    );
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn saturating_sub<T>(a: T, b: T) -> T;
        }

        const GOAL: bool = saturating_sub(5u8, 7) == 0 && saturating_sub(8u8, 4) == 4;
        "#,
        1,
    );
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn saturating_add<T>(a: T, b: T) -> T;
        }

        const GOAL: i8 = saturating_add(5, 8);
        "#,
        13,
    );
}

#[test]
fn allocator() {
    check_number(
        r#"
        //- minicore: sized
        extern "Rust" {
            #[rustc_allocator]
            fn __rust_alloc(size: usize, align: usize) -> *mut u8;
            #[rustc_deallocator]
            fn __rust_dealloc(ptr: *mut u8, size: usize, align: usize);
            #[rustc_reallocator]
            fn __rust_realloc(ptr: *mut u8, old_size: usize, align: usize, new_size: usize) -> *mut u8;
            #[rustc_allocator_zeroed]
            fn __rust_alloc_zeroed(size: usize, align: usize) -> *mut u8;
        }

        const GOAL: u8 = unsafe {
            let ptr = __rust_alloc(4, 1);
            let ptr2 = ((ptr as usize) + 1) as *mut u8;
            *ptr = 23;
            *ptr2 = 32;
            let ptr = __rust_realloc(ptr, 4, 1, 8);
            let ptr = __rust_realloc(ptr, 8, 1, 3);
            let ptr2 = ((ptr as usize) + 1) as *mut u8;
            *ptr + *ptr2
        };
        "#,
        55,
    );
}

#[test]
fn overflowing_add() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn add_with_overflow<T>(x: T, y: T) -> (T, bool);
        }

        const GOAL: u8 = add_with_overflow(1, 2).0;
        "#,
        3,
    );
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn add_with_overflow<T>(x: T, y: T) -> (T, bool);
        }

        const GOAL: u8 = add_with_overflow(1, 2).1 as u8;
        "#,
        0,
    );
}

#[test]
fn needs_drop() {
    check_number(
        r#"
        //- minicore: drop, manually_drop, copy, sized
        use core::mem::ManuallyDrop;
        extern "rust-intrinsic" {
            pub fn needs_drop<T: ?Sized>() -> bool;
        }
        struct X;
        struct NeedsDrop;
        impl Drop for NeedsDrop {
            fn drop(&mut self) {}
        }
        enum Enum<T> {
            A(T),
            B(X),
        }
        const fn val_needs_drop<T>(_v: T) -> bool { needs_drop::<T>() }
        const fn closure_needs_drop() -> bool {
            let a = NeedsDrop;
            let b = X;
            !val_needs_drop(|| &a) && val_needs_drop(move || &a) && !val_needs_drop(move || &b)
        }
        const fn opaque() -> impl Sized {
            || {}
        }
        const fn opaque_copy() -> impl Sized + Copy {
            || {}
        }
        trait Everything {}
        impl<T> Everything for T {}
        const GOAL: bool = !needs_drop::<i32>() && !needs_drop::<X>()
            && needs_drop::<NeedsDrop>() && !needs_drop::<ManuallyDrop<NeedsDrop>>()
            && needs_drop::<[NeedsDrop; 1]>() && !needs_drop::<[NeedsDrop; 0]>()
            && needs_drop::<(X, NeedsDrop)>()
            && needs_drop::<Enum<NeedsDrop>>() && !needs_drop::<Enum<X>>()
            && closure_needs_drop()
            && !val_needs_drop(opaque()) && !val_needs_drop(opaque_copy())
            && needs_drop::<[NeedsDrop]>() && needs_drop::<dyn Everything>()
            && !needs_drop::<&dyn Everything>() && !needs_drop::<str>();
        "#,
        1,
    );
}

#[test]
fn discriminant_value() {
    check_number(
        r#"
        //- minicore: discriminant, option
        use core::marker::DiscriminantKind;
        extern "rust-intrinsic" {
            pub fn discriminant_value<T>(v: &T) -> <T as DiscriminantKind>::Discriminant;
        }
        const GOAL: bool = {
            discriminant_value(&Some(2i32)) == discriminant_value(&Some(5i32))
                && discriminant_value(&Some(2i32)) != discriminant_value(&None::<i32>)
        };
        "#,
        1,
    );
}

#[test]
fn likely() {
    check_number(
        r#"
        #[rustc_intrinsic]
        pub const fn likely(b: bool) -> bool {
            b
        }

        #[rustc_intrinsic]
        pub const fn unlikely(b: bool) -> bool {
            b
        }

        const GOAL: bool = likely(true) && unlikely(true) && !likely(false) && !unlikely(false);
        "#,
        1,
    );
}

#[test]
fn floating_point() {
    // FIXME(#17451): Add `f16` and `f128` tests once intrinsics are added.
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn sqrtf32(x: f32) -> f32;
            pub fn powf32(a: f32, x: f32) -> f32;
            pub fn fmaf32(a: f32, b: f32, c: f32) -> f32;
        }

        const GOAL: f32 = sqrtf32(1.2) + powf32(3.4, 5.6) + fmaf32(-7.8, 1.3, 2.4);
        "#,
        i128::from_le_bytes(pad16(
            &f32::to_le_bytes(1.2f32.sqrt() + 3.4f32.powf(5.6) + (-7.8f32).mul_add(1.3, 2.4)),
            true,
        )),
    );
    #[allow(unknown_lints, clippy::unnecessary_min_or_max)]
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn powif64(a: f64, x: i32) -> f64;
            pub fn sinf64(x: f64) -> f64;
            pub fn minnumf64(x: f64, y: f64) -> f64;
        }

        const GOAL: f64 = powif64(1.2, 5) + sinf64(3.4) + minnumf64(-7.8, 1.3);
        "#,
        i128::from_le_bytes(pad16(
            &f64::to_le_bytes(1.2f64.powi(5) + 3.4f64.sin() + (-7.8f64).min(1.3)),
            true,
        )),
    );
}

#[test]
fn atomic() {
    check_number(
        r#"
        //- minicore: copy
        extern "rust-intrinsic" {
            pub fn atomic_load_seqcst<T: Copy>(src: *const T) -> T;
            pub fn atomic_xchg_acquire<T: Copy>(dst: *mut T, src: T) -> T;
            pub fn atomic_cxchg_release_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
            pub fn atomic_cxchgweak_acquire_acquire<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
            pub fn atomic_store_release<T: Copy>(dst: *mut T, val: T);
            pub fn atomic_xadd_acqrel<T: Copy>(dst: *mut T, src: T) -> T;
            pub fn atomic_xsub_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
            pub fn atomic_and_acquire<T: Copy>(dst: *mut T, src: T) -> T;
            pub fn atomic_nand_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
            pub fn atomic_or_release<T: Copy>(dst: *mut T, src: T) -> T;
            pub fn atomic_xor_seqcst<T: Copy>(dst: *mut T, src: T) -> T;
            pub fn atomic_fence_seqcst();
            pub fn atomic_singlethreadfence_acqrel();
        }

        fn should_not_reach() {
            _ // fails the test if executed
        }

        const GOAL: i32 = {
            let mut x = 5;
            atomic_store_release(&mut x, 10);
            let mut y = atomic_xchg_acquire(&mut x, 100);
            atomic_xadd_acqrel(&mut y, 20);
            if (30, true) != atomic_cxchg_release_seqcst(&mut y, 30, 40) {
                should_not_reach();
            }
            atomic_fence_seqcst();
            if (40, false) != atomic_cxchg_release_seqcst(&mut y, 30, 50) {
                should_not_reach();
            }
            if (40, true) != atomic_cxchgweak_acquire_acquire(&mut y, 40, 30) {
                should_not_reach();
            }
            let mut z = atomic_xsub_seqcst(&mut x, -200);
            atomic_singlethreadfence_acqrel();
            atomic_xor_seqcst(&mut x, 1024);
            atomic_load_seqcst(&x) + z * 3 + atomic_load_seqcst(&y) * 2
        };
        "#,
        660 + 1024,
    );
}

#[test]
fn offset() {
    check_number(
        r#"
        //- minicore: coerce_unsized, index, slice
        extern "rust-intrinsic" {
            pub fn offset<Ptr, Delta>(dst: Ptr, offset: Delta) -> Ptr;
            pub fn arith_offset<T>(dst: *const T, offset: isize) -> *const T;
        }

        const GOAL: i32 = unsafe {
            let ar: &[(i32, i32, i32)] = &[
                (10, 11, 12),
                (20, 21, 22),
                (30, 31, 32),
                (40, 41, 42),
                (50, 51, 52),
            ];
            let ar: *const [(i32, i32, i32)] = ar;
            let ar = ar as *const (i32, i32, i32);
            let element3 = *offset(ar, 2usize);
            let element4 = *arith_offset(ar, 3);
            element3.1 * 100 + element4.0
        };
        "#,
        3140,
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
fn write_bytes() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            fn write_bytes<T>(dst: *mut T, val: u8, count: usize);
        }

        const GOAL: i32 = unsafe {
            let mut x = 2;
            write_bytes(&mut x, 5, 1);
            x
        };
        "#,
        0x05050505,
    );
}

#[test]
fn write_via_move() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            fn write_via_move<T>(ptr: *mut T, value: T);
        }

        const GOAL: i32 = unsafe {
            let mut x = 2;
            write_via_move(&mut x, 100);
            x
        };
        "#,
        100,
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

#[test]
fn ctpop() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn ctpop<T: Copy>(x: T) -> T;
        }

        const GOAL: i64 = ctpop(-29);
        "#,
        61,
    );
}

#[test]
fn ctlz() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn ctlz<T: Copy>(x: T) -> T;
        }

        const GOAL: u8 = ctlz(0b0001_1100_u8);
        "#,
        3,
    );
}

#[test]
fn cttz() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn cttz<T: Copy>(x: T) -> T;
        }

        const GOAL: i64 = cttz(-24);
        "#,
        3,
    );
}

#[test]
fn rotate() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn rotate_left<T: Copy>(x: T, y: T) -> T;
        }

        const GOAL: i64 = rotate_left(0xaa00000000006e1i64, 12);
        "#,
        0x6e10aa,
    );
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn rotate_right<T: Copy>(x: T, y: T) -> T;
        }

        const GOAL: i64 = rotate_right(0x6e10aa, 12);
        "#,
        0xaa00000000006e1,
    );
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn rotate_left<T: Copy>(x: T, y: T) -> T;
        }

        const GOAL: i8 = rotate_left(129, 2);
        "#,
        6,
    );
    check_number(
        r#"
        #[rustc_intrinsic]
        pub fn rotate_right<T: Copy>(x: T, y: T) -> T;

        const GOAL: i32 = rotate_right(10006016, 1020315);
        "#,
        320192512,
    );
}

#[test]
fn simd() {
    check_number(
        r#"
        pub struct i8x16(
            i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,
        );
        #[rustc_intrinsic]
        pub fn simd_bitmask<T, U>(x: T) -> U;
        const GOAL: u16 = simd_bitmask(i8x16(
            0, 1, 0, 0, 2, 255, 100, 0, 50, 0, 1, 1, 0, 0, 0, 0
        ));
        "#,
        0b0000110101110010,
    );
    check_number(
        r#"
        pub struct i8x16(
            i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,i8,
        );
        #[rustc_intrinsic]
        pub fn simd_lt<T, U>(x: T, y: T) -> U;
        #[rustc_intrinsic]
        pub fn simd_bitmask<T, U>(x: T) -> U;
        const GOAL: u16 = simd_bitmask(simd_lt::<i8x16, i8x16>(
            i8x16(
                -105, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            ),
            i8x16(
                -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
            ),
        ));
        "#,
        0xFFFF,
    );
}
