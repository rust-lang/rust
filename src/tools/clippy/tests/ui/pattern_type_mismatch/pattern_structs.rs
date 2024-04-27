#![allow(clippy::all)]
#![warn(clippy::pattern_type_mismatch)]

fn main() {}

fn struct_types() {
    struct Struct<'a> {
        ref_inner: &'a Option<i32>,
    }
    let ref_value = &Struct { ref_inner: &Some(42) };

    // not ok
    let Struct { .. } = ref_value;
    //~^ ERROR: type of pattern does not match the expression type
    if let &Struct { ref_inner: Some(_) } = ref_value {}
    //~^ ERROR: type of pattern does not match the expression type
    if let Struct { ref_inner: Some(_) } = *ref_value {}
    //~^ ERROR: type of pattern does not match the expression type

    // ok
    let &Struct { .. } = ref_value;
    let Struct { .. } = *ref_value;
    if let &Struct { ref_inner: &Some(_) } = ref_value {}
    if let Struct { ref_inner: &Some(_) } = *ref_value {}
}

fn struct_enum_variants() {
    enum StructEnum<'a> {
        Empty,
        Var { inner_ref: &'a Option<i32> },
    }
    let ref_value = &StructEnum::Var { inner_ref: &Some(42) };

    // not ok
    if let StructEnum::Var { .. } = ref_value {}
    //~^ ERROR: type of pattern does not match the expression type
    if let StructEnum::Var { inner_ref: Some(_) } = ref_value {}
    //~^ ERROR: type of pattern does not match the expression type
    if let &StructEnum::Var { inner_ref: Some(_) } = ref_value {}
    //~^ ERROR: type of pattern does not match the expression type
    if let StructEnum::Var { inner_ref: Some(_) } = *ref_value {}
    //~^ ERROR: type of pattern does not match the expression type
    if let StructEnum::Empty = ref_value {}
    //~^ ERROR: type of pattern does not match the expression type

    // ok
    if let &StructEnum::Var { .. } = ref_value {}
    if let StructEnum::Var { .. } = *ref_value {}
    if let &StructEnum::Var { inner_ref: &Some(_) } = ref_value {}
    if let StructEnum::Var { inner_ref: &Some(_) } = *ref_value {}
    if let &StructEnum::Empty = ref_value {}
    if let StructEnum::Empty = *ref_value {}
}
