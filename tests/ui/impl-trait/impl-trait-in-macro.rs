use std::fmt::Debug;

macro_rules! i {
    ($($tr:tt)*) => { impl $($tr)* };
}

fn foo(x: i!(Debug), y: i!(Debug)) -> String {
    let mut a = x;
    a = y; //~ ERROR mismatched
    format!("{:?}", a)
}

trait S<T> {}

fn much_universe<T: S<i!(Debug)>, U: IntoIterator<Item = i!(Iterator<Item = i!(Clone)>)>>(
    _: i!(Debug + Clone),
) {
}

fn main() {}
