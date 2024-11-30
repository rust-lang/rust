//@ check-pass
//@ edition:2018

pub async fn f(x: Option<usize>) {
    x.take();
    //~^ WARNING cannot borrow `x` as mutable, as it is not declared as mutable [E0596]
}

pub async fn g(x: usize) {
    x += 1;
    //~^ WARNING cannot assign twice to immutable variable `x` [E0384]
}

fn main() {}
