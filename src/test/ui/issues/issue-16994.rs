// compile-pass
// skip-codegen
fn cb<'a,T>(_x: Box<Fn((&'a i32, &'a (Vec<&'static i32>, bool))) -> T>) -> T {
    panic!()
}


fn main() {
    cb(Box::new(|(k, &(ref v, b))| (*k, v.clone(), b)));
}
