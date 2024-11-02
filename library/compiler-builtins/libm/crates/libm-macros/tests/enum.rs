#[libm_macros::function_enum(BaseName)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Function {}

#[libm_macros::base_name_enum]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BaseName {}

#[test]
fn as_str() {
    assert_eq!(Function::Sin.as_str(), "sin");
    assert_eq!(Function::Sinf.as_str(), "sinf");
}

#[test]
fn basename() {
    assert_eq!(Function::Sin.base_name(), BaseName::Sin);
    assert_eq!(Function::Sinf.base_name(), BaseName::Sin);
}
