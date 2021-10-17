// run-rustfix
fn foo<T: PartialEq>(a: &T, b: T) {
    let _ = a == b; //~ ERROR can't compare
}

fn main() {
    foo(&1, 1);
}
