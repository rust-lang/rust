#![allow(dead_code)]

trait C {}
impl dyn C { fn f() {} } //~ ERROR duplicate
impl dyn C { fn f() {} }
fn main() { }
