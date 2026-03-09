use super::*;

#[test]
fn test_struct_info_roundtrip() {
    let s = ItemEnum::Struct(Struct {
        generics: Generics { params: vec![], where_predicates: vec![] },
        kind: StructKind::Plain { fields: vec![], has_stripped_fields: false },
        impls: vec![],
    });

    // JSON
    let struct_json = serde_json::to_string(&s).unwrap();
    let de_s = serde_json::from_str(&struct_json).unwrap();
    assert_eq!(s, de_s);

    // Bincode
    let encoded: Vec<u8> = bincode::serialize(&s).unwrap();
    let decoded: ItemEnum = bincode::deserialize(&encoded).unwrap();
    assert_eq!(s, decoded);
}

#[test]
fn test_union_info_roundtrip() {
    let u = ItemEnum::Union(Union {
        generics: Generics { params: vec![], where_predicates: vec![] },
        has_stripped_fields: false,
        fields: vec![],
        impls: vec![],
    });

    // JSON
    let union_json = serde_json::to_string(&u).unwrap();
    let de_u = serde_json::from_str(&union_json).unwrap();
    assert_eq!(u, de_u);

    // Bincode
    let encoded: Vec<u8> = bincode::serialize(&u).unwrap();
    let decoded: ItemEnum = bincode::deserialize(&encoded).unwrap();
    assert_eq!(u, decoded);
}

#[cfg(feature = "rkyv_0_8")]
mod rkyv {
    use std::fmt::Debug;

    use rkyv::Archive;
    use rkyv::api::high::{HighDeserializer, HighSerializer};
    use rkyv::bytecheck::CheckBytes;
    use rkyv::rancor::Strategy;
    use rkyv::ser::allocator::ArenaHandle;
    use rkyv::util::AlignedVec;
    use rkyv::validation::Validator;
    use rkyv::validation::archive::ArchiveValidator;
    use rkyv::validation::shared::SharedValidator;

    use crate::*;

    #[test]
    /// A test to exercise the (de)serialization roundtrip for a representative selection of types,
    /// covering most of the rkyv-specific attributes we had to had.
    fn test_rkyv_roundtrip() {
        // Standard derives: a plain struct and union, mirroring the existing serde/bincode tests.
        let s = ItemEnum::Struct(Struct {
            generics: Generics { params: vec![], where_predicates: vec![] },
            kind: StructKind::Plain { fields: vec![Id(1), Id(2)], has_stripped_fields: false },
            impls: vec![Id(3)],
        });
        rkyv_roundtrip(&s);

        let u = ItemEnum::Union(Union {
            generics: Generics { params: vec![], where_predicates: vec![] },
            has_stripped_fields: false,
            fields: vec![Id(1)],
            impls: vec![],
        });
        rkyv_roundtrip(&u);

        // Extra trait derives, via rkyv(derive(PartialEq, Eq, PartialOrd, Ord, Hash)), on the archived type.
        rkyv_roundtrip(&Id(99));

        // Recursive cycle-breaking: `BorrowedRef` has omit_bounds on its `Box<Type>` field.
        let ty = Type::BorrowedRef {
            lifetime: Some("'a".to_string()),
            is_mutable: false,
            type_: Box::new(Type::Primitive("str".to_string())),
        };
        rkyv_roundtrip(&ty);

        // `Slice` and `Tuple` are tuple-variant fields with omit_bounds on the unnamed field,
        // which required special syntax (attribute inside the parentheses) to compile.
        let ty = Type::Slice(Box::new(Type::Tuple(vec![
            Type::Primitive("u32".to_string()),
            Type::Generic("T".to_string()),
        ])));
        rkyv_roundtrip(&ty);

        // `Path` has serialize_bounds/deserialize_bounds and omit_bounds on its `args` field.
        // `GenericArgs::AngleBracketed` exercises the full recursive chain: `Path` -> `GenericArgs` -> `GenericArg` -> `Type`.
        let path = Path {
            path: "std::option::Option".to_string(),
            id: Id(42),
            args: Some(Box::new(GenericArgs::AngleBracketed {
                args: vec![GenericArg::Type(Type::Primitive("u32".to_string()))],
                constraints: vec![],
            })),
        };
        rkyv_roundtrip(&path);

        // `FunctionPointer` is a `Box<FunctionPointer>` behind `omit_bounds` in `Type`.
        // It transitively contains `Type` via `FunctionSignature`, exercising the cycle from the other direction.
        let fp = Type::FunctionPointer(Box::new(FunctionPointer {
            sig: FunctionSignature {
                inputs: vec![("x".to_string(), Type::Primitive("i32".to_string()))],
                output: Some(Type::Primitive("bool".to_string())),
                is_c_variadic: false,
            },
            generic_params: vec![],
            header: FunctionHeader {
                is_const: false,
                is_unsafe: false,
                is_async: false,
                abi: Abi::Rust,
            },
        }));
        rkyv_roundtrip(&fp);
    }

    /// A helper function for roundtrip testing of rkyv-powered deserialization.
    fn rkyv_roundtrip<T>(value: &T)
    where
        T: PartialEq
            + Debug
            + Archive
            + for<'a> rkyv::Serialize<
                HighSerializer<AlignedVec, ArenaHandle<'a>, rkyv::rancor::Error>,
            >,
        T::Archived: rkyv::Deserialize<T, HighDeserializer<rkyv::rancor::Error>>
            + Debug
            + for<'a> CheckBytes<
                Strategy<Validator<ArchiveValidator<'a>, SharedValidator>, rkyv::rancor::Error>,
            >,
    {
        let bytes =
            rkyv::api::high::to_bytes_in::<_, rkyv::rancor::Error>(value, AlignedVec::new())
                .unwrap();
        let archived = rkyv::api::high::access::<T::Archived, rkyv::rancor::Error>(&bytes)
            .expect("Failed to access archived data");
        let deserialized: T = rkyv::api::deserialize_using::<_, _, rkyv::rancor::Error>(
            archived,
            &mut rkyv::de::Pool::new(),
        )
        .unwrap();
        assert_eq!(value, &deserialized);
    }
}
