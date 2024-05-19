#![allow(clippy::all)]
#![warn(clippy::pattern_type_mismatch)]

fn main() {}

fn alternatives() {
    enum Value<'a> {
        Unused,
        A(&'a Option<i32>),
        B,
    }
    let ref_value = &Value::A(&Some(23));

    // not ok
    if let Value::B | Value::A(_) = ref_value {}
    //~^ ERROR: type of pattern does not match the expression type
    if let &Value::B | &Value::A(Some(_)) = ref_value {}
    //~^ ERROR: type of pattern does not match the expression type
    if let Value::B | Value::A(Some(_)) = *ref_value {}
    //~^ ERROR: type of pattern does not match the expression type

    // ok
    if let &Value::B | &Value::A(_) = ref_value {}
    if let Value::B | Value::A(_) = *ref_value {}
    if let &Value::B | &Value::A(&Some(_)) = ref_value {}
    if let Value::B | Value::A(&Some(_)) = *ref_value {}
}
