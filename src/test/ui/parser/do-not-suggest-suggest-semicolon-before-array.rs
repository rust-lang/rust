fn foo() {}

fn bar() -> [u8; 2] {
    foo()
    [1, 3) //~ ERROR expected one of `.`, `?`, `]`, or an operator, found `,`
}

fn main() {}
