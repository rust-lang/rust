use std::collections::HashSet;

/// natural case from the issue
struct Value(u32);

fn main() {
    let hs = HashSet::<Value>::new();
    hs.insert(Value(0)); //~ ERROR
}

/// synthetic cases
pub struct NoDerives;

struct Object<T>(T);
impl<T: Eq> Object<T> {
    fn use_eq(&self) {}
}
impl<T: Ord> Object<T> {
    fn use_ord(&self) {}
}
impl<T: Ord + PartialOrd> Object<T> {
    fn use_ord_and_partial_ord(&self) {}
}

fn function(foo: Object<NoDerives>) {
    foo.use_eq(); //~ ERROR
    foo.use_ord(); //~ ERROR
    foo.use_ord_and_partial_ord(); //~ ERROR
}
