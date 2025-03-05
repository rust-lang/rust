// Test for the basic function of parallel front end
//
//@ parallel-front-end-robustness
//@ compile-flags: -Z threads=8
//@ run-pass

fn main() {
    println!("Hello world!");
}
