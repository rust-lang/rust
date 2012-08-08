trait foo {
    fn foo() -> int;
}

impl ~[uint]: foo {
    fn foo() -> int {1} //~ NOTE candidate #1 is `__extensions__::foo`
}

impl ~[int]: foo {
    fn foo() -> int {2} //~ NOTE candidate #2 is `__extensions__::foo`
}

fn main() {
    let x = ~[];
    x.foo(); //~ ERROR multiple applicable methods in scope
}
