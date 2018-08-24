// issue #12418

#![deny(unused_unsafe)]

fn main() {
    unsafe { println!("foo"); } //~ ERROR unnecessary `unsafe`
}
