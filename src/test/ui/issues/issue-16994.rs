#![feature(rustc_attrs)]

fn cb<'a,T>(_x: Box<Fn((&'a i32, &'a (Vec<&'static i32>, bool))) -> T>) -> T {
    panic!()
}

#[rustc_error]
fn main() { //~ ERROR compilation successful
    cb(Box::new(|(k, &(ref v, b))| (*k, v.clone(), b)));
}
