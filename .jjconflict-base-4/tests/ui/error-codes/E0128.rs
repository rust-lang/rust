struct Foo<T=U, U=()> { //~ ERROR E0128
    field1: T,
    field2: U,
}

fn main() {
}
