extern crate hygiene_example_codegen;

pub use hygiene_example_codegen::hello;

pub fn print(string: &str) {
    println!("{}", string);
}
