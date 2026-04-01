fn foo<T>() -> T {
    panic!()
}

fn main() {
    foo(); //~ ERROR type annotations needed
}
