// run-pass
#![allow(dead_code)]
// aux-build:png2.rs
// compile-flags:--extern png2
// edition:2018

mod png {
    use png2 as png_ext;

    fn foo() -> png_ext::DecodingError { unimplemented!() }
}

fn main() {
    println!("Hello, world!");
}
