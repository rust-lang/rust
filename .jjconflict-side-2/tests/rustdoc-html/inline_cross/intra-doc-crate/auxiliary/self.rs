#![crate_name = "cross_crate_self"]

/// Link to [Self]
/// Link to [crate]
pub struct S;

impl S {
    /// Link to [Self::f]
    pub fn f() {}
}
