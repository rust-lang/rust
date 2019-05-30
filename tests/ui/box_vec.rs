#![warn(clippy::all)]
#![allow(clippy::boxed_local, clippy::needless_pass_by_value)]
#![allow(clippy::blacklisted_name)]

macro_rules! boxit {
    ($init:expr, $x:ty) => {
        let _: Box<$x> = Box::new($init);
    };
}

fn test_macro() {
    boxit!(Vec::new(), Vec<u8>);
}
pub fn test(foo: Box<Vec<bool>>) {
    println!("{:?}", foo.get(0))
}

pub fn test2(foo: Box<dyn Fn(Vec<u32>)>) {
    // pass if #31 is fixed
    foo(vec![1, 2, 3])
}

pub fn test_local_not_linted() {
    let _: Box<Vec<bool>>;
}

fn main() {
    test(Box::new(Vec::new()));
    test2(Box::new(|v| println!("{:?}", v)));
    test_macro();
    test_local_not_linted();
}
