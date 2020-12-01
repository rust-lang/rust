#![crate_name = "private"]
// compile-flags: --document-private-items
/// docs [DontDocMe]
// @has private/struct.DocMe.html '//*a[@href="../private/struct.DontDocMe.html"]' 'DontDocMe'
pub struct DocMe;
struct DontDocMe;
