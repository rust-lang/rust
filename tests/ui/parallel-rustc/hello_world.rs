// Test for the basic function of parallel front end
//
//@ compile-flags: -Z threads=8
//@ run-pass
//@ compare-output-by-lines

fn main() {
    println!("Hello world!");
}
