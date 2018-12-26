#![allow(dead_code)]

trait C {}
impl C { fn f() {} } //~ ERROR duplicate
impl C { fn f() {} }
fn main() { }
