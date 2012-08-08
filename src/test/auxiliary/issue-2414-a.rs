#[link(name = "a", vers = "0.1")];
#[crate_type = "lib"];

type t1 = uint;

trait foo {
    fn foo();
}

impl ~str: foo {
    fn foo() {}
}

