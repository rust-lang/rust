const A: usize = *&5;

fn foo() -> usize {
    A
}

fn main() {
    assert_eq!(foo(), A);
}
