use core::any::TypeId;
use core::intrinsics::assume;

#[test]
fn test_typeid_sized_types() {
    struct X;
    struct Y(#[allow(dead_code)] u32);

    assert_eq!(TypeId::of::<X>(), TypeId::of::<X>());
    assert_eq!(TypeId::of::<Y>(), TypeId::of::<Y>());
    assert!(TypeId::of::<X>() != TypeId::of::<Y>());
}

#[test]
fn test_typeid_unsized_types() {
    trait Z {}
    struct X(#[allow(dead_code)] str);
    struct Y(#[allow(dead_code)] dyn Z + 'static);

    assert_eq!(TypeId::of::<X>(), TypeId::of::<X>());
    assert_eq!(TypeId::of::<Y>(), TypeId::of::<Y>());
    assert!(TypeId::of::<X>() != TypeId::of::<Y>());
}

// Check that `const_assume` feature allow `assume` intrinsic
// to be used in const contexts.
#[test]
fn test_assume_can_be_in_const_contexts() {
    const unsafe fn foo(x: usize, y: usize) -> usize {
        // SAFETY: the entire function is not safe,
        // but it is just an example not used elsewhere.
        unsafe { assume(y != 0) };
        x / y
    }
    let rs = unsafe { foo(42, 97) };
    assert_eq!(rs, 0);
}

#[test]
const fn test_write_bytes_in_const_contexts() {
    use core::intrinsics::write_bytes;

    const TEST: [u32; 3] = {
        let mut arr = [1u32, 2, 3];
        unsafe {
            write_bytes(arr.as_mut_ptr(), 0, 2);
        }
        arr
    };

    assert!(TEST[0] == 0);
    assert!(TEST[1] == 0);
    assert!(TEST[2] == 3);

    const TEST2: [u32; 3] = {
        let mut arr = [1u32, 2, 3];
        unsafe {
            write_bytes(arr.as_mut_ptr(), 1, 2);
        }
        arr
    };

    assert!(TEST2[0] == 16843009);
    assert!(TEST2[1] == 16843009);
    assert!(TEST2[2] == 3);
}

#[test]
fn test_hints_in_const_contexts() {
    use core::intrinsics::{likely, unlikely};

    // In const contexts, they just return their argument.
    const {
        assert!(true == likely(true));
        assert!(false == likely(false));
        assert!(true == unlikely(true));
        assert!(false == unlikely(false));
        assert!(42u32 == core::intrinsics::black_box(42u32));
        assert!(42u32 == core::hint::black_box(42u32));
    }
}

#[test]
fn test_const_allocate_at_runtime() {
    use core::intrinsics::const_allocate;
    unsafe {
        assert!(const_allocate(4, 4).is_null());
    }
}

#[test]
fn test_const_deallocate_at_runtime() {
    use core::intrinsics::const_deallocate;
    const X: &u32 = &42u32;
    let x = &0u32;
    unsafe {
        const_deallocate(X as *const _ as *mut u8, 4, 4); // nop
        const_deallocate(x as *const _ as *mut u8, 4, 4); // nop
        const_deallocate(core::ptr::null_mut(), 1, 1); // nop
    }
}

#[test]
fn test_three_way_compare_in_const_contexts() {
    use core::cmp::Ordering::{self, *};
    use core::intrinsics::three_way_compare;

    const UNSIGNED_LESS: Ordering = three_way_compare(123_u16, 456);
    const UNSIGNED_EQUAL: Ordering = three_way_compare(456_u16, 456);
    const UNSIGNED_GREATER: Ordering = three_way_compare(789_u16, 456);
    const CHAR_LESS: Ordering = three_way_compare('A', 'B');
    const CHAR_EQUAL: Ordering = three_way_compare('B', 'B');
    const CHAR_GREATER: Ordering = three_way_compare('C', 'B');
    const SIGNED_LESS: Ordering = three_way_compare(123_i64, 456);
    const SIGNED_EQUAL: Ordering = three_way_compare(456_i64, 456);
    const SIGNED_GREATER: Ordering = three_way_compare(789_i64, 456);

    assert_eq!(UNSIGNED_LESS, Less);
    assert_eq!(UNSIGNED_EQUAL, Equal);
    assert_eq!(UNSIGNED_GREATER, Greater);
    assert_eq!(CHAR_LESS, Less);
    assert_eq!(CHAR_EQUAL, Equal);
    assert_eq!(CHAR_GREATER, Greater);
    assert_eq!(SIGNED_LESS, Less);
    assert_eq!(SIGNED_EQUAL, Equal);
    assert_eq!(SIGNED_GREATER, Greater);
}

fn fallback_cma<T: core::intrinsics::fallback::CarryingMulAdd>(
    a: T,
    b: T,
    c: T,
    d: T,
) -> (T::Unsigned, T) {
    a.carrying_mul_add(b, c, d)
}

#[test]
fn carrying_mul_add_fallback_u32() {
    let r = fallback_cma::<u32>(0x9e37_79b9, 0x7f4a_7c15, 0xf39c_c060, 0x5ced_c834);
    assert_eq!(r, (0x2087_20c1, 0x4eab_8e1d));
    let r = fallback_cma::<u32>(0x1082_276b, 0xf3a2_7251, 0xf86c_6a11, 0xd0c1_8e95);
    assert_eq!(r, (0x7aa0_1781, 0x0fb6_0528));
}

#[test]
fn carrying_mul_add_fallback_i32() {
    let r = fallback_cma::<i32>(-1, -1, -1, -1);
    assert_eq!(r, (u32::MAX, -1));
    let r = fallback_cma::<i32>(1, -1, 1, 1);
    assert_eq!(r, (1, 0));
}

#[test]
fn carrying_mul_add_fallback_u128() {
    assert_eq!(fallback_cma::<u128>(1, 1, 1, 1), (3, 0));
    assert_eq!(fallback_cma::<u128>(0, 0, u128::MAX, u128::MAX), (u128::MAX - 1, 1));
    assert_eq!(
        fallback_cma::<u128>(u128::MAX, u128::MAX, u128::MAX, u128::MAX),
        (u128::MAX, u128::MAX),
    );

    let r = fallback_cma::<u128>(
        0x243f6a8885a308d313198a2e03707344,
        0xa4093822299f31d0082efa98ec4e6c89,
        0x452821e638d01377be5466cf34e90c6c,
        0xc0ac29b7c97c50dd3f84d5b5b5470917,
    );
    assert_eq!(r, (0x8050ec20ed554e40338d277e00b674e7, 0x1739ee6cea07da409182d003859b59d8));
    let r = fallback_cma::<u128>(
        0x9216d5d98979fb1bd1310ba698dfb5ac,
        0x2ffd72dbd01adfb7b8e1afed6a267e96,
        0xba7c9045f12c7f9924a19947b3916cf7,
        0x0801f2e2858efc16636920d871574e69,
    );
    assert_eq!(r, (0x185525545fdb2fefb502a3a602efd628, 0x1b62d35fe3bff6b566f99667ef7ebfd6));
}

#[test]
fn carrying_mul_add_fallback_i128() {
    let r = fallback_cma::<i128>(-1, -1, -1, -1);
    assert_eq!(r, (u128::MAX, -1));
    let r = fallback_cma::<i128>(1, -1, 1, 1);
    assert_eq!(r, (1, 0));
}
