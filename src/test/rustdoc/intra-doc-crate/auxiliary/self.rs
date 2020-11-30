#![crate_name = "cross_crate_self"]
pub struct S;

impl S {
    /// Link to [Self::f]
    pub fn f() {}
}
