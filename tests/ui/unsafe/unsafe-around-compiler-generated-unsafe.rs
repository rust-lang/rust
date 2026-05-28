//@ edition:2018

#![deny(unused_unsafe)]

fn main() {
    let _ = async {
        unsafe { async {}.await; } //~ ERROR unnecessary `unsafe`
    };

    // `format_args!` expands with a compiler-generated unsafe block
    unsafe { println!("foo"); } //~ ERROR unnecessary `unsafe`
}
