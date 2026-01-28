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
fn test_slices() {
    match const { Type::of::<[usize]>() }.kind {
        TypeKind::Slice(slice) => assert_eq!(slice.element_ty, TypeId::of::<usize>()),
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

    // Mutable references.
    match const { Type::of::<&mut u64>() }.kind {
        TypeKind::Reference(reference) => {
            assert_eq!(reference.pointee, TypeId::of::<u64>());
            assert!(reference.mutable);
        }
        _ => unreachable!(),
    }

    // Wide references.
    match const { Type::of::<&dyn Any>() }.kind {
        TypeKind::Reference(reference) => {
            assert_eq!(reference.pointee, TypeId::of::<dyn Any>());
            assert!(!reference.mutable);
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_pointers() {
    // Immutable pointer.
    match const { Type::of::<*const u8>() }.kind {
        TypeKind::Pointer(pointer) => {
            assert_eq!(pointer.pointee, TypeId::of::<u8>());
            assert!(!pointer.mutable);
        }
        _ => unreachable!(),
    }

    // Mutable pointer.
    match const { Type::of::<*mut u64>() }.kind {
        TypeKind::Pointer(pointer) => {
            assert_eq!(pointer.pointee, TypeId::of::<u64>());
            assert!(pointer.mutable);
        }
        _ => unreachable!(),
    }

    // Wide pointer.
    match const { Type::of::<*const dyn Any>() }.kind {
        TypeKind::Pointer(pointer) => {
            assert_eq!(pointer.pointee, TypeId::of::<dyn Any>());
            assert!(!pointer.mutable);
        }
        _ => unreachable!(),
    }
}

#[test]
fn test_dynamic_traits() {
    use std::collections::HashSet;
    use std::mem::type_info::DynTraitPredicate;
    trait A<T> {}

    trait B<const CONST_NUM: i32> {
        type Foo;
    }

    trait FooTrait<'a, 'b, const CONST_NUM: i32> {}

    trait ProjectorTrait<'a, 'b> {}

    fn preds_of<T: ?Sized + 'static>() -> &'static [DynTraitPredicate] {
        match const { Type::of::<T>() }.kind {
            TypeKind::DynTrait(d) => d.predicates,
            _ => unreachable!(),
        }
    }

    fn pred<'a>(preds: &'a [DynTraitPredicate], want: TypeId) -> &'a DynTraitPredicate {
        preds
            .iter()
            .find(|p| p.trait_ty.ty == want)
            .unwrap_or_else(|| panic!("missing predicate for {want:?}"))
    }

    fn assert_typeid_set_eq(actual: &[TypeId], expected: &[TypeId]) {
        let actual_set: HashSet<TypeId> = actual.iter().copied().collect();
        let expected_set: HashSet<TypeId> = expected.iter().copied().collect();
        assert_eq!(actual.len(), actual_set.len(), "duplicates present: {actual:?}");
        assert_eq!(
            actual_set, expected_set,
            "unexpected ids.\nactual: {actual:?}\nexpected: {expected:?}"
        );
    }

    fn assert_predicates_exact(preds: &[DynTraitPredicate], expected_pred_ids: &[TypeId]) {
        let actual_pred_ids: Vec<TypeId> = preds.iter().map(|p| p.trait_ty.ty).collect();
        assert_typeid_set_eq(&actual_pred_ids, expected_pred_ids);
    }

    // dyn Send
    {
        let preds = preds_of::<dyn Send>();
        assert_predicates_exact(preds, &[TypeId::of::<dyn Send>()]);

        let p = pred(preds, TypeId::of::<dyn Send>());
        assert!(p.trait_ty.is_auto);
    }

    // dyn A<i32>
    {
        let preds = preds_of::<dyn A<i32>>();
        assert_predicates_exact(preds, &[TypeId::of::<dyn A<i32>>()]);

        let p = pred(preds, TypeId::of::<dyn A<i32>>());
        assert!(!p.trait_ty.is_auto);
    }

    // dyn B<5, Foo = i32>
    {
        let preds = preds_of::<dyn B<5, Foo = i32>>();
        assert_predicates_exact(preds, &[TypeId::of::<dyn B<5, Foo = i32>>()]);

        let e = pred(preds, TypeId::of::<dyn B<5, Foo = i32>>());
        assert!(!e.trait_ty.is_auto);
    }

    // dyn for<'a> FooTrait<'a, 'a, 7>
    {
        let preds = preds_of::<dyn for<'a> FooTrait<'a, 'a, 7>>();
        assert_predicates_exact(preds, &[TypeId::of::<dyn for<'a> FooTrait<'a, 'a, 7>>()]);

        let foo = pred(preds, TypeId::of::<dyn for<'a> FooTrait<'a, 'a, 7>>());
        assert!(!foo.trait_ty.is_auto);
    }

    // dyn FooTrait<'static, 'static, 7>
    {
        let preds = preds_of::<dyn FooTrait<'static, 'static, 7>>();
        assert_predicates_exact(preds, &[TypeId::of::<dyn FooTrait<'static, 'static, 7>>()]);

        let foo = pred(preds, TypeId::of::<dyn FooTrait<'static, 'static, 7>>());
        assert!(!foo.trait_ty.is_auto);
    }

    // dyn for<'a, 'b> FooTrait<'a, 'b, 7>
    {
        let preds = preds_of::<dyn for<'a, 'b> FooTrait<'a, 'b, 7>>();
        assert_predicates_exact(preds, &[TypeId::of::<dyn for<'a, 'b> FooTrait<'a, 'b, 7>>()]);

        let foo = pred(preds, TypeId::of::<dyn for<'a, 'b> FooTrait<'a, 'b, 7>>());
        assert!(!foo.trait_ty.is_auto);
    }

    // dyn for<'a, 'b> ProjectorTrait<'a, 'b>
    {
        let preds = preds_of::<dyn for<'a, 'b> ProjectorTrait<'a, 'b>>();
        assert_predicates_exact(preds, &[TypeId::of::<dyn for<'a, 'b> ProjectorTrait<'a, 'b>>()]);

        let proj = pred(preds, TypeId::of::<dyn for<'a, 'b> ProjectorTrait<'a, 'b>>());
        assert!(!proj.trait_ty.is_auto);
    }

    // dyn for<'a> FooTrait<'a, 'a, 7> + Send + Sync
    {
        let preds = preds_of::<dyn for<'a> FooTrait<'a, 'a, 7> + Send + Sync>();
        assert_predicates_exact(
            preds,
            &[
                TypeId::of::<dyn for<'a> FooTrait<'a, 'a, 7>>(),
                TypeId::of::<dyn Send>(),
                TypeId::of::<dyn Sync>(),
            ],
        );

        let foo = pred(preds, TypeId::of::<dyn for<'a> FooTrait<'a, 'a, 7>>());
        assert!(!foo.trait_ty.is_auto);

        let send = pred(preds, TypeId::of::<dyn Send>());
        assert!(send.trait_ty.is_auto);

        let sync = pred(preds, TypeId::of::<dyn Sync>());
        assert!(sync.trait_ty.is_auto);
    }
}
