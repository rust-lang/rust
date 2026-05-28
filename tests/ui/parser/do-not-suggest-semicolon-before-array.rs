fn foo() {}

fn bar() -> [u8; 2] {
    foo()
    [1, 3) //~ ERROR mismatched closing delimiter
}

fn main() {}
