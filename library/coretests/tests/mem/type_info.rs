#![allow(dead_code)]

use std::any::{Any, TypeId};
use std::mem::offset_of;
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

    const {
        let ty_id = TypeId::of::<()>();
        assert!(ty_id.size() == Some(size_of::<()>()));
        assert!(ty_id.variants() == 1);
        assert!(ty_id.fields(0) == 0);

        let ty_id = TypeId::of::<(u8,)>();
        assert!(ty_id.size() == Some(size_of::<(u8,)>()));
        assert!(ty_id.variants() == 1);
        assert!(ty_id.fields(0) == 1);
        assert!(ty_id.field(0, 0).type_id() == TypeId::of::<u8>());

        let ty_id = TypeId::of::<(u8, u16)>();
        assert!(ty_id.size() == Some(size_of::<(u8, u16)>()));
        assert!(ty_id.variants() == 1);
        assert!(ty_id.fields(0) == 2);
        assert!(ty_id.field(0, 0).type_id() == TypeId::of::<u8>());
        assert!(ty_id.field(0, 1).type_id() == TypeId::of::<u16>());
    }
}

#[test]
fn test_structs() {
    use TypeKind::*;

    const {
        struct TestStruct {
            first: u8,
            second: u16,
            reference: &'static u16,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<TestStruct>() else { panic!() };
        assert!(!ty.non_exhaustive);
        assert!(ty.fields.len() == 3);
        assert!(ty.fields[0].name == "first");
        assert!(ty.fields[0].ty == TypeId::of::<u8>());
        assert!(ty.fields[0].offset == offset_of!(TestStruct, first));
        assert!(ty.fields[1].name == "second");
        assert!(ty.fields[1].ty == TypeId::of::<u16>());
        assert!(ty.fields[1].offset == offset_of!(TestStruct, second));
        assert!(ty.fields[2].name == "reference");
        assert!(ty.fields[2].ty == TypeId::of::<&'static u16>());
        assert!(ty.fields[2].offset == offset_of!(TestStruct, reference));

        let ty_id = TypeId::of::<TestStruct>();
        assert!(ty_id.size() == Some(size_of::<TestStruct>()));
        assert!(ty_id.variants() == 1);
        assert!(ty_id.fields(0) == 3);
        assert!(ty_id.field(0, 0).type_id() == TypeId::of::<u8>());
        assert!(ty_id.field(0, 1).type_id() == TypeId::of::<u16>());
        assert!(ty_id.field(0, 2).type_id() == TypeId::of::<&u16>());
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

        let ty_id = TypeId::of::<TupleStruct>();
        assert!(ty_id.size() == Some(size_of::<TupleStruct>()));
        assert!(ty_id.variants() == 1);
        assert!(ty_id.fields(0) == 2);
        assert!(ty_id.field(0, 0).type_id() == TypeId::of::<u8>());
        assert!(ty_id.field(0, 1).type_id() == TypeId::of::<u16>());
    }

    const {
        struct Generics<'a, T, const C: u64> {
            a: &'a T,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<Generics<'static, i32, 1_u64>>() else {
            panic!()
        };
        assert!(ty.fields.len() == 1);
        assert!(ty.generics.len() == 3);

        let Generic::Lifetime(_) = ty.generics[0] else { panic!() };
        let Generic::Type(GenericType { ty: generic_ty, .. }) = ty.generics[1] else { panic!() };
        assert!(generic_ty == TypeId::of::<i32>());
        let Generic::Const(Const { ty: const_ty, .. }) = ty.generics[2] else { panic!() };
        assert!(const_ty == TypeId::of::<u64>());
    }
}

#[test]
fn test_unions() {
    use TypeKind::*;

    const {
        union TestUnion {
            first: i16,
            second: u16,
        }

        let Type { kind: Union(ty), .. } = Type::of::<TestUnion>() else { panic!() };
        assert!(ty.fields.len() == 2);
        assert!(ty.fields[0].name == "first");
        assert!(ty.fields[0].offset == offset_of!(TestUnion, first));
        assert!(ty.fields[1].name == "second");
        assert!(ty.fields[1].offset == offset_of!(TestUnion, second));

        let ty_id = TypeId::of::<TestUnion>();
        assert!(ty_id.size() == Some(size_of::<TestUnion>()));
        assert!(ty_id.variants() == 1);
        assert!(ty_id.fields(0) == 2);
        assert!(ty_id.field(0, 0).type_id() == TypeId::of::<i16>());
        assert!(ty_id.field(0, 1).type_id() == TypeId::of::<u16>());
    }

    const {
        union Generics<'a, T: Copy, const C: u64> {
            a: T,
            z: &'a (),
        }

        let Type { kind: Union(ty), .. } = Type::of::<Generics<'static, i32, 1_u64>>() else {
            panic!()
        };
        assert!(ty.fields.len() == 2);
        assert!(ty.fields[0].offset == offset_of!(Generics<'static, i32, 1_u64>, a));
        assert!(ty.fields[1].offset == offset_of!(Generics<'static, i32, 1_u64>, z));

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

        let Type { kind: Enum(ty), .. } = Type::of::<E>() else { panic!() };
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

        let ty_id = TypeId::of::<E>();
        assert!(ty_id.size() == Some(size_of::<E>()));
        assert!(ty_id.variants() == 3);
        assert!(ty_id.fields(0) == 1);
        assert!(ty_id.fields(1) == 0);
        assert!(ty_id.fields(2) == 2);
        assert!(ty_id.field(0, 0).type_id() == TypeId::of::<u32>());
        assert!(ty_id.field(2, 0).type_id() == TypeId::of::<()>());
        assert!(ty_id.field(2, 1).type_id() == TypeId::of::<&str>());
    }

    const {
        let Type { kind: Enum(ty), .. } = Type::of::<Option<i32>>() else { panic!() };
        assert!(ty.variants.len() == 2);
        assert!(ty.generics.len() == 1);
        let Generic::Type(GenericType { ty: generic_ty, .. }) = ty.generics[0] else { panic!() };
        assert!(generic_ty == TypeId::of::<i32>());

        let ty_id = TypeId::of::<Option<i32>>();
        assert!(ty_id.size() == Some(size_of::<Option<i32>>()));
        assert!(ty_id.variants() == 2);
        assert!(ty_id.fields(0) == 0);
        assert!(ty_id.fields(1) == 1);
        assert!(ty_id.field(1, 0).type_id() == TypeId::of::<i32>());
    }
}

#[test]
fn test_primitives() {
    use TypeKind::*;

    const {
        let Type { kind: Bool(_ty), .. } = (const { Type::of::<bool>() }) else { panic!() };
        let ty_id = TypeId::of::<bool>();
        assert!(ty_id.size() == Some(size_of::<bool>()));
        assert!(ty_id.variants() == 1);

        let Type { kind: Char(_ty), .. } = (const { Type::of::<char>() }) else { panic!() };
        let ty_id = TypeId::of::<char>();
        assert!(ty_id.size() == Some(size_of::<char>()));
        assert!(ty_id.variants() == 1);

        let Type { kind: Int(ty), .. } = (const { Type::of::<i32>() }) else { panic!() };
        assert!(ty.bits == 32);
        assert!(ty.signed);
        let ty_id = TypeId::of::<i32>();
        assert!(ty_id.size() == Some(size_of::<i32>()));
        assert!(ty_id.variants() == 1);

        let Type { kind: Int(ty), .. } = (const { Type::of::<isize>() }) else { panic!() };
        assert!(ty.bits as usize == size_of::<isize>() * 8);
        assert!(ty.signed);
        let ty_id = TypeId::of::<isize>();
        assert!(ty_id.size() == Some(size_of::<isize>()));
        assert!(ty_id.variants() == 1);

        let Type { kind: Int(ty), .. } = (const { Type::of::<u32>() }) else { panic!() };
        assert!(ty.bits == 32);
        assert!(!ty.signed);
        let ty_id = TypeId::of::<u32>();
        assert!(ty_id.size() == Some(size_of::<u32>()));
        assert!(ty_id.variants() == 1);

        let Type { kind: Int(ty), .. } = (const { Type::of::<usize>() }) else { panic!() };
        assert!(ty.bits as usize == size_of::<usize>() * 8);
        assert!(!ty.signed);
        let ty_id = TypeId::of::<usize>();
        assert!(ty_id.size() == Some(size_of::<usize>()));
        assert!(ty_id.variants() == 1);

        let Type { kind: Float(ty), .. } = (const { Type::of::<f32>() }) else { panic!() };
        assert!(ty.bits == 32);
        let ty_id = TypeId::of::<f32>();
        assert!(ty_id.size() == Some(size_of::<f32>()));
        assert!(ty_id.variants() == 1);

        let Type { kind: Str(_ty), .. } = (const { Type::of::<str>() }) else { panic!() };
        let ty_id = TypeId::of::<str>();
        assert!(ty_id.size() == None);
        assert!(ty_id.variants() == 1);
    }
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

#[test]
fn test_attributes() {
    use TypeKind::*;

    const {
        #[allow(dead_code)]
        struct AttrStruct {
            #[allow(dead_code)]
            field_a: u8,
            field_b: u16,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<AttrStruct>() else { panic!() };
        assert!(ty.attributes.len() == 1);
        assert!(ty.attributes[0].path == "allow");
        assert!(ty.attributes[0].args == "dead_code");

        assert!(ty.fields[0].attributes.len() == 1);
        assert!(ty.fields[0].attributes[0].path == "allow");
        assert!(ty.fields[0].attributes[0].args == "dead_code");

        assert!(ty.fields[1].attributes.len() == 0);
    }

    const {
        #[allow(dead_code)]
        enum AttrEnum {
            #[allow(unused)]
            VariantA(u32),
            VariantB,
        }

        let Type { kind: Enum(ty), .. } = Type::of::<AttrEnum>() else { panic!() };
        assert!(ty.attributes.len() == 1);
        assert!(ty.attributes[0].path == "allow");

        assert!(ty.variants[0].attributes.len() == 1);
        assert!(ty.variants[0].attributes[0].path == "allow");
        assert!(ty.variants[0].attributes[0].args == "unused");

        assert!(ty.variants[1].attributes.len() == 0);
    }

    const {
        struct NoAttrs {
            x: u8,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<NoAttrs>() else { panic!() };
        assert!(ty.attributes.len() == 0);
        assert!(ty.fields[0].attributes.len() == 0);
    }

    const {
        #[allow(dead_code)]
        union AttrUnion {
            #[allow(dead_code)]
            a: u32,
            b: f32,
        }

        let Type { kind: Union(ty), .. } = Type::of::<AttrUnion>() else { panic!() };
        assert!(ty.attributes.len() == 1);
        assert!(ty.attributes[0].path == "allow");
        assert!(ty.attributes[0].args == "dead_code");

        assert!(ty.fields[0].attributes.len() == 1);
        assert!(ty.fields[0].attributes[0].path == "allow");
        assert!(ty.fields[0].attributes[0].args == "dead_code");

        assert!(ty.fields[1].attributes.len() == 0);
    }

    const {
        #[allow(dead_code)]
        #[allow(unused)]
        struct MultiAttr {
            x: u8,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<MultiAttr>() else { panic!() };
        assert!(ty.attributes.len() == 2);
        assert!(ty.attributes[0].path == "allow");
        assert!(ty.attributes[0].args == "dead_code");
        assert!(ty.attributes[1].path == "allow");
        assert!(ty.attributes[1].args == "unused");
    }

    const {
        #[non_exhaustive]
        struct EmptyArgs {
            x: u8,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<EmptyArgs>() else { panic!() };
        // #[non_exhaustive] is a parsed built-in attribute, not reflected here.
        assert!(ty.non_exhaustive);
        assert!(ty.attributes.len() == 0);
    }

    const {
        #[allow(dead_code)]
        enum FieldAttrEnum {
            Variant {
                #[allow(unused)]
                a: u32,
                b: u8,
            },
        }

        let Type { kind: Enum(ty), .. } = Type::of::<FieldAttrEnum>() else { panic!() };
        assert!(ty.variants[0].fields[0].attributes.len() == 1);
        assert!(ty.variants[0].fields[0].attributes[0].path == "allow");
        assert!(ty.variants[0].fields[0].attributes[0].args == "unused");
        assert!(ty.variants[0].fields[1].attributes.len() == 0);
    }

    const {
        #[deny(missing_docs)]
        #[allow(dead_code)]
        #[warn(unused)]
        struct LayeredAttrs {
            #[allow(dead_code, unused_variables)]
            #[deny(unused_imports)]
            field_a: u32,

            #[allow(dead_code)]
            #[warn(unused_assignments)]
            field_b: u8,

            #[forbid(unsafe_code)]
            #[allow(dead_code)]
            field_c: u16,

            #[allow(dead_code)]
            field_d: u64,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<LayeredAttrs>() else { panic!() };
        assert!(ty.attributes.len() == 3);
        assert!(ty.attributes[0].path == "deny");
        assert!(ty.attributes[0].args == "missing_docs");
        assert!(ty.attributes[1].path == "allow");
        assert!(ty.attributes[1].args == "dead_code");
        assert!(ty.attributes[2].path == "warn");
        assert!(ty.attributes[2].args == "unused");

        assert!(ty.fields[0].attributes.len() == 2);
        assert!(ty.fields[0].attributes[0].path == "allow");
        assert!(ty.fields[0].attributes[0].args == "dead_code, unused_variables");
        assert!(ty.fields[0].attributes[1].path == "deny");
        assert!(ty.fields[0].attributes[1].args == "unused_imports");

        assert!(ty.fields[1].attributes.len() == 2);
        assert!(ty.fields[1].attributes[0].path == "allow");
        assert!(ty.fields[1].attributes[0].args == "dead_code");
        assert!(ty.fields[1].attributes[1].path == "warn");
        assert!(ty.fields[1].attributes[1].args == "unused_assignments");

        assert!(ty.fields[2].attributes.len() == 2);
        assert!(ty.fields[2].attributes[0].path == "forbid");
        assert!(ty.fields[2].attributes[0].args == "unsafe_code");
        assert!(ty.fields[2].attributes[1].path == "allow");
        assert!(ty.fields[2].attributes[1].args == "dead_code");

        assert!(ty.fields[3].attributes.len() == 1);
        assert!(ty.fields[3].attributes[0].path == "allow");
        assert!(ty.fields[3].attributes[0].args == "dead_code");
    }

    const {
        #[allow(dead_code)]
        #[deny(unused)]
        enum LayeredAttrsEnum {
            #[allow(dead_code)]
            #[warn(unused_variables)]
            VariantA {
                #[forbid(unsafe_code)]
                x: u32,
                y: u8,
            },
            #[allow(dead_code)]
            VariantB,
            #[allow(unused)]
            VariantC(u8),
        }

        let Type { kind: Enum(ty), .. } = Type::of::<LayeredAttrsEnum>() else { panic!() };
        assert!(ty.attributes.len() == 2);
        assert!(ty.attributes[0].path == "allow");
        assert!(ty.attributes[0].args == "dead_code");
        assert!(ty.attributes[1].path == "deny");
        assert!(ty.attributes[1].args == "unused");

        assert!(ty.variants[0].attributes.len() == 2);
        assert!(ty.variants[0].attributes[0].path == "allow");
        assert!(ty.variants[0].attributes[0].args == "dead_code");
        assert!(ty.variants[0].attributes[1].path == "warn");
        assert!(ty.variants[0].attributes[1].args == "unused_variables");
        assert!(ty.variants[0].fields[0].attributes.len() == 1);
        assert!(ty.variants[0].fields[0].attributes[0].path == "forbid");
        assert!(ty.variants[0].fields[0].attributes[0].args == "unsafe_code");
        assert!(ty.variants[0].fields[1].attributes.len() == 0);

        assert!(ty.variants[1].attributes.len() == 1);
        assert!(ty.variants[1].attributes[0].path == "allow");
        assert!(ty.variants[1].attributes[0].args == "dead_code");
        assert!(ty.variants[1].fields.len() == 0);

        assert!(ty.variants[2].attributes.len() == 1);
        assert!(ty.variants[2].attributes[0].path == "allow");
        assert!(ty.variants[2].attributes[0].args == "unused");
        assert!(ty.variants[2].fields[0].attributes.len() == 0);
    }

    const {
        #[non_exhaustive]
        #[allow(dead_code)]
        #[deny(unused)]
        struct MixedAttrs {
            #[allow(dead_code)]
            x: u8,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<MixedAttrs>() else { panic!() };
        assert!(ty.non_exhaustive);
        assert!(ty.attributes.len() == 2);
        assert!(ty.attributes[0].path == "allow");
        assert!(ty.attributes[0].args == "dead_code");
        assert!(ty.attributes[1].path == "deny");
        assert!(ty.attributes[1].args == "unused");
    }

    const {
        #[rustfmt::skip]
        struct NamespacedAttr {
            x: u8,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<NamespacedAttr>() else { panic!() };
        assert!(ty.attributes.len() == 1);
        assert!(ty.attributes[0].path == "rustfmt::skip");
        assert!(ty.attributes[0].args == "");
    }

    const {
        struct TupleFieldAttrs(#[allow(dead_code)] u8, u16);

        let Type { kind: Struct(ty), .. } = Type::of::<TupleFieldAttrs>() else { panic!() };
        assert!(ty.fields.len() == 2);
        assert!(ty.fields[0].attributes.len() == 1);
        assert!(ty.fields[0].attributes[0].path == "allow");
        assert!(ty.fields[0].attributes[0].args == "dead_code");
        assert!(ty.fields[1].attributes.len() == 0);
    }

    const {
        let Type { kind: Enum(ty), .. } = Type::of::<Option<i32>>() else { panic!() };
        let _ = ty.attributes;
        let _ = ty.variants[0].attributes;
        let _ = ty.variants[1].attributes;
    }

    const {
        #[cfg_attr(test, allow(dead_code))]
        struct CfgAttrTest {
            x: u8,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<CfgAttrTest>() else { panic!() };
        assert!(ty.attributes.len() == 1);
        assert!(ty.attributes[0].path == "allow");
        assert!(ty.attributes[0].args == "dead_code");
    }

    const {
        #[allow(unused_attributes)]
        #[allow()]
        struct EmptyDelim {
            x: u8,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<EmptyDelim>() else { panic!() };
        assert!(ty.attributes.len() == 2);
        assert!(ty.attributes[0].path == "allow");
        assert!(ty.attributes[0].args == "unused_attributes");
        assert!(ty.attributes[1].path == "allow");
        assert!(ty.attributes[1].args == "");
    }

    const {
        #[rustfmt::skip(some::path)]
        struct PathArg {
            x: u8,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<PathArg>() else { panic!() };
        assert!(ty.attributes.len() == 1);
        assert!(ty.attributes[0].path == "rustfmt::skip");
        assert!(ty.attributes[0].args == "some::path");
    }

    const {
        #[rustfmt::skip(a::b, c::d)]
        struct MultiPathArg {
            x: u8,
        }

        let Type { kind: Struct(ty), .. } = Type::of::<MultiPathArg>() else { panic!() };
        assert!(ty.attributes.len() == 1);
        assert!(ty.attributes[0].path == "rustfmt::skip");
        assert!(ty.attributes[0].args == "a::b, c::d");
    }
}
