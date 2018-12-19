struct S {
    x: usize,
    y: usize,
}

fn main() {
    S { x: 4,
        y: 5 };
}

fn foo() { //~ ERROR this file contains an un-closed delimiter
