#![allow(dead_code)]

use std::any::{Any, TypeId};
use std::mem::type_info::{Const, Generic, GenericType, Type, TypeKind};

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

    const {
        struct TestStruct {
            first: u8,
            second: u16,
        }

        let Type { kind: Struct(ty), size, .. } = Type::of::<TestStruct>() else { panic!() };
        assert!(size == Some(size_of::<TestStruct>()));
        assert!(!ty.non_exhaustive);
        assert!(ty.fields.len() == 2);
        assert!(ty.fields[0].name == "first");
        assert!(ty.fields[1].name == "second");
    }

    const {
        #[non_exhaustive]
        struct NonExhaustive {
            a: u8,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<NonExhaustive>() else { panic!() };
        assert!(ty.non_exhaustive);
    }

    const {
        struct TupleStruct(u8, u16);

        let Type { kind: Struct(ty), .. } = Type::of::<TupleStruct>() else { panic!() };
        assert!(ty.fields.len() == 2);
        assert!(ty.fields[0].name == "0");
        assert!(ty.fields[0].ty == TypeId::of::<u8>());
        assert!(ty.fields[1].name == "1");
        assert!(ty.fields[1].ty == TypeId::of::<u16>());
    }

    const {
        struct Generics<'a, T, const C: u64> {
            a: T,
            z: &'a (), // FIXME(type_info): offset of this field is dumped as 0, which may not be correct
        }

        let Type { kind: Struct(ty), .. } = Type::of::<Generics<'static, i32, 1_u64>>() else {
            panic!()
        };
        assert!(ty.fields.len() == 2);
        assert!(ty.generics.len() == 3);

        let Generic::Lifetime(_) = ty.generics[0] else { panic!() };
        let Generic::Type(GenericType { ty: generic_ty, .. }) = ty.generics[1] else { panic!() };
        assert!(generic_ty == TypeId::of::<i32>());
        let Generic::Const(Const { ty: const_ty, .. }) = ty.generics[2] else { panic!() };
        assert!(const_ty == TypeId::of::<u64>());
    }
}

#[test]
fn test_enums() {
    use TypeKind::*;

    const {
        enum E {
            Some(u32),
            None,
            #[non_exhaustive]
            Foomp {
                a: (),
                b: &'static str,
            },
        }

        let Type { kind: Enum(ty), size, .. } = Type::of::<E>() else { panic!() };
        assert!(size == Some(size_of::<E>()));
        assert!(ty.variants.len() == 3);

        assert!(ty.variants[0].name == "Some");
        assert!(!ty.variants[0].non_exhaustive);
        assert!(ty.variants[0].fields.len() == 1);

        assert!(ty.variants[1].name == "None");
        assert!(!ty.variants[1].non_exhaustive);
        assert!(ty.variants[1].fields.len() == 0);

        assert!(ty.variants[2].name == "Foomp");
        assert!(ty.variants[2].non_exhaustive);
        assert!(ty.variants[2].fields.len() == 2);
    }

    const {
        let Type { kind: Enum(ty), size, .. } = Type::of::<Option<i32>>() else { panic!() };
        assert!(size == Some(size_of::<Option<i32>>()));
        assert!(ty.variants.len() == 2);
        assert!(ty.generics.len() == 1);
        let Generic::Type(GenericType { ty: generic_ty, .. }) = ty.generics[0] else { panic!() };
        assert!(generic_ty == TypeId::of::<i32>());
    }
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
