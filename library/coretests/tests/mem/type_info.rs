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
                        assert!(a.bit_width == 8 && a.signed);
                        assert!(b.bit_width == 8 && !b.signed);
                    }
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
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
    assert_eq!(ty.bit_width, 32);
    assert!(ty.signed);

    let Type { kind: Int(ty), size, .. } = (const { Type::of::<isize>() }) else { panic!() };
    assert_eq!(size, Some(size_of::<isize>()));
    assert_eq!(ty.bit_width, size_of::<isize>() * 8);
    assert!(ty.signed);

    let Type { kind: Int(ty), size, .. } = (const { Type::of::<u32>() }) else { panic!() };
    assert_eq!(size, Some(4));
    assert_eq!(ty.bit_width, 32);
    assert!(!ty.signed);

    let Type { kind: Int(ty), size, .. } = (const { Type::of::<usize>() }) else { panic!() };
    assert_eq!(size, Some(size_of::<usize>()));
    assert_eq!(ty.bit_width, size_of::<usize>() * 8);
    assert!(!ty.signed);

    let Type { kind: Float(ty), size, .. } = (const { Type::of::<f32>() }) else { panic!() };
    assert_eq!(size, Some(4));
    assert_eq!(ty.bit_width, 32);

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
