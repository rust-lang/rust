use core::any::TypeId;
use core::intrinsics::assume;

#[test]
fn test_typeid_sized_types() {
    struct X;
    struct Y(u32);

    assert_eq!(TypeId::of::<X>(), TypeId::of::<X>());
    assert_eq!(TypeId::of::<Y>(), TypeId::of::<Y>());
    assert!(TypeId::of::<X>() != TypeId::of::<Y>());
}

#[test]
fn test_typeid_unsized_types() {
    trait Z {}
    struct X(str);
    struct Y(dyn Z + 'static);

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

#[cfg(not(bootstrap))]
#[test]
fn test_const_allocate_at_runtime() {
    use core::intrinsics::const_allocate;
    unsafe {
        assert!(const_allocate(4, 4).is_null());
    }
}

#[cfg(not(bootstrap))]
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
