use std::any::TypeId;
use std::mem::type_info::{Abi, FnPtr, Type, TypeKind};

const STRING_TY: TypeId = const { TypeId::of::<String>() };
const U8_TY: TypeId = const { TypeId::of::<u8>() };
const UNIT_TY: TypeId = const { TypeId::of::<()>() };
const TUPLE_STRING_U8_TY: TypeId = const { TypeId::of::<(String, u8)>() };

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
fn test_typekind() {
    assert!(matches!(const { Type::of::<fn()>().kind }, TypeKind::FnPtr));
    assert!(matches!(const { Type::of::<fn(&u8, &u8)>().kind }, TypeKind::FnPtr));
    assert!(matches!(const { Type::of::<fn(unsafe fn())>().kind }, TypeKind::FnPtr));
    assert!(matches!(const { Type::of::<fn(extern "system" fn())>().kind }, TypeKind::FnPtr));
    assert!(matches!(
        const { Type::of::<fn(#[rustc_splat] (String, u8))>().kind },
        TypeKind::FnPtr
    ));
}

#[test]
fn test_ref() {
    // references are tricky because the lifetimes give the references different type ids
    // so we check the pointees instead
    const F: FnPtr = TypeId::of::<fn(&u8, &u8)>().function_ptr().unwrap();
    assert_eq!(const { F.inputs()[0].points_to() }, Some(U8_TY));
    assert_eq!(const { F.inputs()[1].points_to() }, Some(U8_TY));
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
