use std::{num::ParseFloatError, str::FromStr};

// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.f64.html#method.from_str"]' 'f64::from_str()'
/// Uses the rules from [`f64::from_str()`].
pub struct Number {
    pub value: f64,
}
