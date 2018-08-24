#![feature(const_fn)]

const X : usize = 2;

const fn f(x: usize) -> usize {
    let mut sum = 0;
    //~^ let bindings in constant functions are unstable
    //~| statements in constant functions are unstable
    for i in 0..x {
        //~^ ERROR E0015
        //~| ERROR E0019
        sum += i;
    }
    sum
}

#[allow(unused_variables)]
fn main() {
    let a : [i32; f(X)]; //~ ERROR E0080
}
