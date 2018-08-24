fn foo<T: PartialEq>(a: &T, b: T) {
    a == b; //~ ERROR E0277
}

fn main() {
    foo(&1, 1);
}
