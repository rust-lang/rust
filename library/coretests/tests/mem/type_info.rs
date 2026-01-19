use std::any::{Any, TypeId};
use std::mem::type_info::{Type, TypeKind};

#[test]
fn test_arrays() {
    // Normal array.
    match const { Type::of::<[u16; 4]>() }.kind {
        TypeKind::Array(array) => {
            assert_eq!(array.element_ty, TypeId::of::<u16>());
            assert_eq!(array.len, 4);
        }
        _ => unreachable!(),
    }

    // Zero-length array.
    match const { Type::of::<[bool; 0]>() }.kind {
        TypeKind::Array(array) => {
            assert_eq!(array.element_ty, TypeId::of::<bool>());
            assert_eq!(array.len, 0);
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_tuples() {
    fn assert_tuple_arity<T: 'static, const N: usize>() {
        match const { Type::of::<T>() }.kind {
            TypeKind::Tuple(tup) => {
                assert_eq!(tup.fields.len(), N);
            }
            _ => unreachable!(),
        }
    }

    assert_tuple_arity::<(), 0>();
    assert_tuple_arity::<(u8,), 1>();
    assert_tuple_arity::<(u8, u8), 2>();

    const {
        match Type::of::<(i8, u8)>().kind {
            TypeKind::Tuple(tup) => {
                let [a, b] = tup.fields else { unreachable!() };

                assert!(a.offset == 0);
                assert!(b.offset == 1);

                match (a.ty.info().kind, b.ty.info().kind) {
                    (TypeKind::Int(a), TypeKind::Int(b)) => {
                        assert!(a.bits == 8 && a.signed);
                        assert!(b.bits == 8 && !b.signed);
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn test_structs() {
    use TypeKind::*;

    struct TestStruct {
        first: u8,
        second: u16,
    }
    #[non_exhaustive]
    struct NonExhaustive {
        a: u8,
    }
    struct TupleStruct(u8, u16);
    struct Generics<'a, A, B, const C: u64> {
        a: A,
        b: B,
        l: &'a (), // FIXME(type_info): offset of this field is dumped as 0, which may not be correct
    }

    let Type { kind: Struct(ty), size, .. } = (const { Type::of::<TestStruct>() }) else {
        panic!()
    };
    assert_eq!(size, Some(size_of::<TestStruct>()));
    assert!(!ty.non_exhaustive);
    assert_eq!(ty.fields.len(), 2);
    assert_eq!(ty.fields[0].name, "first");
    assert_eq!(ty.fields[1].name, "second");

    let Type { kind: Struct(ty), size, .. } = (const { Type::of::<NonExhaustive>() }) else {
        panic!()
    };
    assert_eq!(size, Some(1));
    assert!(ty.non_exhaustive);

    let Type { kind: Struct(ty), size, .. } = (const { Type::of::<TupleStruct>() }) else {
        panic!()
    };
    assert_eq!(ty.fields.len(), 2);
    assert_eq!(ty.fields[0].name, "0");
    assert_eq!(ty.fields[1].name, "1");

    let Type { kind: Struct(ty), size, .. } =
        (const { Type::of::<Generics<'static, i32, u32, 1>>() })
    else {
        panic!()
    };
    assert_eq!(ty.fields.len(), 3);
    let Generic::Lifetime(_) = ty.generics[0] else { panic!() };
    let Generic::Type(GenericType { ty: generic_ty }) = ty.generics[1] else { panic!() };
    let TypeKind::Int(generic_ty) = generic_ty.info().kind else { panic!() };
    assert_eq!(generic_ty.bits, 32);
    let Generic::Type(GenericType { ty: generic_ty }) = ty.generics[2] else { panic!() };
    let TypeKind::Int(generic_ty) = generic_ty.info().kind else { panic!() };
    assert_eq!(generic_ty.bits, 32);
}

#[test]
fn test_primitives() {
    use TypeKind::*;

    let Type { kind: Bool(_ty), size, .. } = (const { Type::of::<bool>() }) else { panic!() };
    assert_eq!(size, Some(1));

    let Type { kind: Char(_ty), size, .. } = (const { Type::of::<char>() }) else { panic!() };
    assert_eq!(size, Some(4));

    let Type { kind: Int(ty), size, .. } = (const { Type::of::<i32>() }) else { panic!() };
    assert_eq!(size, Some(4));
    assert_eq!(ty.bits, 32);
    assert!(ty.signed);

    let Type { kind: Int(ty), size, .. } = (const { Type::of::<isize>() }) else { panic!() };
    assert_eq!(size, Some(size_of::<isize>()));
    assert_eq!(ty.bits as usize, size_of::<isize>() * 8);
    assert!(ty.signed);

    let Type { kind: Int(ty), size, .. } = (const { Type::of::<u32>() }) else { panic!() };
    assert_eq!(size, Some(4));
    assert_eq!(ty.bits, 32);
    assert!(!ty.signed);

    let Type { kind: Int(ty), size, .. } = (const { Type::of::<usize>() }) else { panic!() };
    assert_eq!(size, Some(size_of::<usize>()));
    assert_eq!(ty.bits as usize, size_of::<usize>() * 8);
    assert!(!ty.signed);

    let Type { kind: Float(ty), size, .. } = (const { Type::of::<f32>() }) else { panic!() };
    assert_eq!(size, Some(4));
    assert_eq!(ty.bits, 32);

    let Type { kind: Str(_ty), size, .. } = (const { Type::of::<str>() }) else { panic!() };
    assert_eq!(size, None);
}

#[test]
fn test_references() {
    // Immutable reference.
    match const { Type::of::<&u8>() }.kind {
        TypeKind::Reference(reference) => {
            assert_eq!(reference.pointee, TypeId::of::<u8>());
            assert!(!reference.mutable);
        }
        _ => unreachable!(),
    }

    // Mutable pointer.
    match const { Type::of::<&mut u64>() }.kind {
        TypeKind::Reference(reference) => {
            assert_eq!(reference.pointee, TypeId::of::<u64>());
            assert!(reference.mutable);
        }
        _ => unreachable!(),
    }

    // Wide pointer.
    match const { Type::of::<&dyn Any>() }.kind {
        TypeKind::Reference(reference) => {
            assert_eq!(reference.pointee, TypeId::of::<dyn Any>());
            assert!(!reference.mutable);
        }
        _ => unreachable!(),
    }
}
