fn foo<const X: usize>() -> usize {
    0
}

fn main() {
    foo(); //~ ERROR type annotations needed
}
