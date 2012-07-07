iface foo { fn foo(); }

impl of foo for uint {} //~ ERROR missing method `foo`

impl of foo for uint { fn foo() -> int {} } //~ ERROR incompatible type

impl of int for uint { fn foo() {} } //~ ERROR can only implement interface

fn main() {}
