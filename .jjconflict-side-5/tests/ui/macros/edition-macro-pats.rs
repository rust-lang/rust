//@ run-pass
//@ edition:2021

macro_rules! foo {
    (a $x:pat_param) => {};
    (b $x:pat) => {};
}

fn main() {
    foo!(a None);
    foo!(b 1 | 2);
}
