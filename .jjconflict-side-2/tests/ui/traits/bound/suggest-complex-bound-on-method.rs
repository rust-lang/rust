//@ run-rustfix
#![allow(dead_code)]
struct Application;
// https://github.com/rust-lang/rust/issues/144734
trait Trait {
    type Error: std::error::Error;

    fn run(&self) -> Result<(), Self::Error>;
}

#[derive(Debug)]
enum ApplicationError {
    Quit,
}

impl Application {
    fn thing<T: Trait>(&self, t: T) -> Result<(), ApplicationError> {
        t.run()?; //~ ERROR E0277
        Ok(())
    }
}

fn main() {}
