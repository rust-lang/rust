#[libm_macros::function_enum(BaseName)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Identifier {}

#[libm_macros::base_name_enum]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BaseName {}

#[test]
fn as_str() {
    assert_eq!(Identifier::Sin.as_str(), "sin");
    assert_eq!(Identifier::Sinf.as_str(), "sinf");
}

#[test]
fn from_str() {
    assert_eq!(Identifier::from_str("sin").unwrap(), Identifier::Sin);
    assert_eq!(Identifier::from_str("sinf").unwrap(), Identifier::Sinf);
}

#[test]
fn basename() {
    assert_eq!(Identifier::Sin.base_name(), BaseName::Sin);
    assert_eq!(Identifier::Sinf.base_name(), BaseName::Sin);
}

#[test]
fn math_op() {
    assert_eq!(Identifier::Sin.math_op().float_ty, FloatTy::F64);
    assert_eq!(Identifier::Sinf.math_op().float_ty, FloatTy::F32);
}

// Replicate the structure that we have in `libm-test`
mod op {
    include!("../../libm-macros/src/shared.rs");
}

use op::FloatTy;
