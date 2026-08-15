#![deny(unused_allocation)]

fn main() {
    _ = Box::new([1]).len(); //~ error: unnecessary allocation, use `&` instead
}
