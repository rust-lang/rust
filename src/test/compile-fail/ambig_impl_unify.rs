impl methods for ~[uint] {
    fn foo() -> int {1} //! NOTE candidate #1 is `methods::foo`
}

impl methods for ~[int] {
    fn foo() -> int {2} //! NOTE candidate #2 is `methods::foo`
}

fn main() {
    let x = ~[];
    x.foo(); //! ERROR multiple applicable methods in scope
}