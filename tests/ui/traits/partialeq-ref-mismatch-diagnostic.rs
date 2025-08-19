//! Check diagnostic messages for `PartialEq` trait bound mismatches between `&T` and `T`.

fn foo<T: PartialEq>(a: &T, b: T) {
    a == b; //~ ERROR E0277
}

fn foo2<T: PartialEq>(a: &T, b: T) {
    a == b; //~ ERROR E0277
}

fn main() {
    foo(&1, 1);
    foo2(&1, 1);
}
