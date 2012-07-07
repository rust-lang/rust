iface foo { fn foo(); }

impl of foo for uint {}

impl of foo for uint { fn foo() -> int {} }

impl of int for uint { fn foo() {} } //~ ERROR interface

fn main() {}
