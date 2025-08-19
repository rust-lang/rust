#![deny(clippy::derive_deserialize_allowing_unknown)]

use serde::{Deserialize, Deserializer};

#[derive(Deserialize)] //~ derive_deserialize_allowing_unknown
struct Struct {
    flag: bool,
    limit: u64,
}

#[derive(Deserialize)] //~ derive_deserialize_allowing_unknown
enum Enum {
    A(bool),
    B { limit: u64 },
}

// negative tests

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructWithDenyUnknownFields {
    flag: bool,
    limit: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
enum EnumWithDenyUnknownFields {
    A(bool),
    B { limit: u64 },
}

#[derive(Deserialize)]
#[serde(untagged, deny_unknown_fields)]
enum MultipleSerdeAttributes {
    A(bool),
    B { limit: u64 },
}

#[derive(Deserialize)]
struct TupleStruct(u64, bool);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
enum EnumWithOnlyTupleVariants {
    A(bool),
    B(u64),
}

struct ManualSerdeImplementation;

impl<'de> Deserialize<'de> for ManualSerdeImplementation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let () = <() as Deserialize>::deserialize(deserializer)?;
        Ok(ManualSerdeImplementation)
    }
}
