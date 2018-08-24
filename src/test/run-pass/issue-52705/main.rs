// aux-build:png.rs
// edition:2018

mod png {
    use png as png_ext;

    fn foo() -> png_ext::DecodingError { unimplemented!() }
}

fn main() {
    println!("Hello, world!");
}
