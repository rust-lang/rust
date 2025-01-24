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
