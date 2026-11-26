use std::any::TypeId;
use std::mem::type_info::{Abi, TypeKind};

const STRING_TY: TypeId = const { TypeId::of::<String>() };
const U8_TY: TypeId = const { TypeId::of::<u8>() };
const UNIT_TY: TypeId = const { TypeId::of::<()>() };
const TUPLE_STRING_U8_TY: TypeId = const { TypeId::of::<(String, u8)>() };
const STATIC_REGION_UNIT_TY: TypeId = const { TypeId::of::<&'static ()>() };

#[test]
fn test_fn_ptrs() {
    let f = const { TypeId::of::<fn()>().function_ptr().unwrap() };
    assert_eq!(f.is_unsafe(), false);
    assert_eq!(f.abi(), Abi::ExternRust);
    assert_eq!(f.inputs(), &[]);
    assert_eq!(f.output(), UNIT_TY);
    assert_eq!(f.is_variadic(), false);
    assert_eq!(f.splatted(), None);
}

#[test]
fn test_type() {
    assert!(matches!(const { TypeKind::of::<fn()>() }, TypeKind::FnPtr));
    assert!(matches!(const { TypeKind::of::<fn(&u8, &u8)>() }, TypeKind::FnPtr));
    assert!(matches!(const { TypeKind::of::<fn(unsafe fn())>() }, TypeKind::FnPtr));
    assert!(matches!(const { TypeKind::of::<fn(extern "system" fn())>() }, TypeKind::FnPtr));
    assert!(matches!(const { TypeKind::of::<fn(#[rustc_splat] (String, u8))>() }, TypeKind::FnPtr));
}

#[test]
fn test_ref() {
    // function_ptr replaces bound regions with erased ones.
    // So, check if TypeId of &'a () is identical to &'static ()
    let f1 = const { TypeId::of::<for<'a> fn(&'a ()) -> &'a ()>().function_ptr().unwrap() };
    assert_eq!(f1.inputs(), [STATIC_REGION_UNIT_TY]);
    assert_eq!(f1.output(), STATIC_REGION_UNIT_TY);

    // function_ptr() does not replace inner bound regions of a nested fn
    let f2 =
        const { TypeId::of::<for<'a> fn(for<'x> fn(&'x u8, &'a u8))>().function_ptr().unwrap() };
    assert_eq!(f2.inputs(), [const { TypeId::of::<for<'x> fn(&'x u8, &'static u8)>() }]);
}

#[test]
fn test_unsafe() {
    assert_eq!(const { TypeId::of::<unsafe fn()>().function_ptr() }.unwrap().is_unsafe(), true);
}

#[test]
fn test_abi() {
    assert_eq!(
        const { TypeId::of::<extern "Rust" fn()>().function_ptr() }.unwrap().abi(),
        Abi::ExternRust
    );

    assert_eq!(
        const { TypeId::of::<extern "C" fn()>().function_ptr() }.unwrap().abi(),
        Abi::ExternC
    );

    assert_eq!(
        const { TypeId::of::<unsafe extern "system" fn()>().function_ptr() }.unwrap().abi(),
        Abi::Named("system")
    );
}

#[test]
fn test_inputs() {
    assert_eq!(
        const { TypeId::of::<fn(String, u8)>().function_ptr() }.unwrap().inputs(),
        [STRING_TY, U8_TY]
    );

    assert_eq!(
        const { TypeId::of::<fn(val: String, p2: u8)>().function_ptr() }.unwrap().inputs(),
        [STRING_TY, U8_TY]
    );
}

#[test]
fn test_output() {
    let f = const { TypeId::of::<fn() -> u8>().function_ptr() }.unwrap();
    assert_eq!(f.output(), U8_TY);
}

#[test]
fn test_variadic() {
    let f = const { TypeId::of::<extern "C" fn(u8, ...)>().function_ptr() }.unwrap();
    assert_eq!(f.abi(), Abi::ExternC);
    assert_eq!(f.inputs(), [U8_TY]);
    assert_eq!(f.is_variadic(), true);
}

#[test]
fn test_splat() {
    let f = const { TypeId::of::<fn(#[rustc_splat] (String, u8))>().function_ptr() }.unwrap();
    assert_eq!(f.inputs(), [TUPLE_STRING_U8_TY]);
    assert_eq!(f.splatted(), Some(0));
}

#[test]
fn test_not_splat() {
    let f = const { TypeId::of::<fn((String, u8))>().function_ptr() }.unwrap();
    assert_eq!(f.inputs(), [TUPLE_STRING_U8_TY]);
    assert_eq!(f.splatted(), None);
}
