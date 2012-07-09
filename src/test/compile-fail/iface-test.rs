iface foo { fn foo(); }

impl of int for uint { fn foo() {} } //~ ERROR interface

fn main() {}
