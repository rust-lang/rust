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

// The memory used by a `Crate` can get large. These types are the ones that
// contribute the most to its size.
#[test]
#[cfg(target_pointer_width = "64")]
fn test_type_sizes() {
    // tidy-alphabetical-start
    assert_eq!(size_of::<AssocItemConstraint>(), 112);
    assert_eq!(size_of::<Crate>(), 184);
    assert_eq!(size_of::<ExternalCrate>(), 48);
    assert_eq!(size_of::<FunctionPointer>(), 168);
    assert_eq!(size_of::<GenericArg>(), 80);
    assert_eq!(size_of::<GenericArgs>(), 104);
    assert_eq!(size_of::<GenericBound>(), 72);
    assert_eq!(size_of::<GenericParamDef>(), 136);
    assert_eq!(size_of::<Impl>(), 304);
    assert_eq!(size_of::<Item>(), 552);
    assert_eq!(size_of::<ItemSummary>(), 32);
    assert_eq!(size_of::<PolyTrait>(), 64);
    assert_eq!(size_of::<PreciseCapturingArg>(), 32);
    assert_eq!(size_of::<TargetFeature>(), 80);
    assert_eq!(size_of::<Type>(), 80);
    assert_eq!(size_of::<WherePredicate>(), 160);
    // tidy-alphabetical-end
}
