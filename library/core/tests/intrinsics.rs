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
