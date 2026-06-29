use std::any::TypeId;
use std::mem::type_info::{Abi, FnPtr, Type, TypeKind};

const STRING_TY: TypeId = const { TypeId::of::<String>() };
const U8_TY: TypeId = const { TypeId::of::<u8>() };
const _U8_REF_TY: TypeId = const { TypeId::of::<&u8>() };
const UNIT_TY: TypeId = const { TypeId::of::<()>() };
const TUPLE_STRING_U8_TY: TypeId = const { TypeId::of::<(String, u8)>() };

#[test]
fn test_fn_ptrs() {
    let TypeKind::FnPtr(FnPtr {
        unsafety: false,
        abi: Abi::ExternRust,
        inputs: &[],
        output,
        variadic: false,
        is_splatted: false,
        splatted_index: _,
    }) = (const { Type::of::<fn()>().kind })
    else {
        panic!();
    };
    assert_eq!(output, UNIT_TY);
}
#[test]
fn test_ref() {
    const {
        // references are tricky because the lifetimes give the references different type ids
        // so we check the pointees instead
        let TypeKind::FnPtr(FnPtr {
            unsafety: false,
            abi: Abi::ExternRust,
            inputs: &[ty1, ty2],
            output,
            variadic: false,
            is_splatted: false,
            splatted_index: _,
        }) = (const { Type::of::<fn(&u8, &u8)>().kind })
        else {
            panic!();
        };
        if output != UNIT_TY {
            panic!();
        }
        let TypeKind::Reference(reference) = ty1.info().kind else {
            panic!();
        };
        if reference.pointee != U8_TY {
            panic!();
        }
        let TypeKind::Reference(reference) = ty2.info().kind else {
            panic!();
        };
        if reference.pointee != U8_TY {
            panic!();
        }
    }
}

#[test]
fn test_unsafe() {
    let TypeKind::FnPtr(FnPtr {
        unsafety: true,
        abi: Abi::ExternRust,
        inputs: &[],
        output,
        variadic: false,
        is_splatted: false,
        splatted_index: _,
    }) = (const { Type::of::<unsafe fn()>().kind })
    else {
        panic!();
    };
    assert_eq!(output, UNIT_TY);
}
#[test]
fn test_abi() {
    let TypeKind::FnPtr(FnPtr {
        unsafety: false,
        abi: Abi::ExternRust,
        inputs: &[],
        output,
        variadic: false,
        is_splatted: false,
        splatted_index: _,
    }) = (const { Type::of::<extern "Rust" fn()>().kind })
    else {
        panic!();
    };
    assert_eq!(output, UNIT_TY);

    let TypeKind::FnPtr(FnPtr {
        unsafety: false,
        abi: Abi::ExternC,
        inputs: &[],
        output,
        variadic: false,
        is_splatted: false,
        splatted_index: _,
    }) = (const { Type::of::<extern "C" fn()>().kind })
    else {
        panic!();
    };
    assert_eq!(output, UNIT_TY);

    let TypeKind::FnPtr(FnPtr {
        unsafety: true,
        abi: Abi::Named("system"),
        inputs: &[],
        output,
        variadic: false,
        is_splatted: false,
        splatted_index: _,
    }) = (const { Type::of::<unsafe extern "system" fn()>().kind })
    else {
        panic!();
    };
    assert_eq!(output, UNIT_TY);
}

#[test]
fn test_inputs() {
    let TypeKind::FnPtr(FnPtr {
        unsafety: false,
        abi: Abi::ExternRust,
        inputs: &[ty1, ty2],
        output,
        variadic: false,
        is_splatted: false,
        splatted_index: _,
    }) = (const { Type::of::<fn(String, u8)>().kind })
    else {
        panic!();
    };
    assert_eq!(output, UNIT_TY);
    assert_eq!(ty1, STRING_TY);
    assert_eq!(ty2, U8_TY);

    let TypeKind::FnPtr(FnPtr {
        unsafety: false,
        abi: Abi::ExternRust,
        inputs: &[ty1, ty2],
        output,
        variadic: false,
        is_splatted: false,
        splatted_index: _,
    }) = (const { Type::of::<fn(val: String, p2: u8)>().kind })
    else {
        panic!();
    };
    assert_eq!(output, UNIT_TY);
    assert_eq!(ty1, STRING_TY);
    assert_eq!(ty2, U8_TY);
}

#[test]
fn test_output() {
    let TypeKind::FnPtr(FnPtr {
        unsafety: false,
        abi: Abi::ExternRust,
        inputs: &[],
        output,
        variadic: false,
        is_splatted: false,
        splatted_index: _,
    }) = (const { Type::of::<fn() -> u8>().kind })
    else {
        panic!();
    };
    assert_eq!(output, U8_TY);
}

#[test]
fn test_variadic() {
    let TypeKind::FnPtr(FnPtr {
        unsafety: false,
        abi: Abi::ExternC,
        inputs: [ty1],
        output,
        variadic: true,
        is_splatted: false,
        splatted_index: _,
    }) = &(const { Type::of::<extern "C" fn(u8, ...)>().kind })
    else {
        panic!();
    };
    assert_eq!(output, &UNIT_TY);
    assert_eq!(*ty1, U8_TY);
}

#[test]
fn test_splat() {
    #[rustfmt::skip]
    let TypeKind::FnPtr(fn_ptr_ty) = &(const { Type::of::<fn(#[splat] (String, u8))>().kind }) else {
        panic!();
    };
    let FnPtr {
        unsafety: false,
        abi: Abi::ExternRust,
        inputs: [ty1],
        output,
        variadic: false,
        is_splatted: true,
        splatted_index: 0,
    } = fn_ptr_ty
    else {
        panic!();
    };
    assert_eq!(output, &UNIT_TY);
    assert_eq!(*ty1, TUPLE_STRING_U8_TY);
    assert_eq!(fn_ptr_ty.splatted(), Some(0));
}

#[test]
fn test_not_splat() {
    let TypeKind::FnPtr(fn_ptr_ty) = &(const { Type::of::<fn((String, u8))>().kind }) else {
        panic!();
    };
    let FnPtr {
        unsafety: false,
        abi: Abi::ExternRust,
        inputs: [ty1],
        output,
        variadic: false,
        is_splatted: false,
        splatted_index: _,
    } = fn_ptr_ty
    else {
        panic!();
    };
    assert_eq!(output, &UNIT_TY);
    assert_eq!(*ty1, TUPLE_STRING_U8_TY);
    assert_eq!(fn_ptr_ty.splatted(), None);
}
