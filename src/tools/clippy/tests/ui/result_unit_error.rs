#![warn(clippy::result_unit_err)]

pub fn returns_unit_error() -> Result<u32, ()> {
    //~^ ERROR: this returns a `Result<_, ()>`
    Err(())
}

fn private_unit_errors() -> Result<String, ()> {
    Err(())
}

pub trait HasUnitError {
    fn get_that_error(&self) -> Result<bool, ()>;
    //~^ ERROR: this returns a `Result<_, ()>`

    fn get_this_one_too(&self) -> Result<bool, ()> {
        //~^ ERROR: this returns a `Result<_, ()>`
        Err(())
    }
}

impl HasUnitError for () {
    fn get_that_error(&self) -> Result<bool, ()> {
        Ok(true)
    }
}

trait PrivateUnitError {
    fn no_problem(&self) -> Result<usize, ()>;
}

pub struct UnitErrorHolder;

impl UnitErrorHolder {
    pub fn unit_error(&self) -> Result<usize, ()> {
        //~^ ERROR: this returns a `Result<_, ()>`
        Ok(0)
    }
}

// https://github.com/rust-lang/rust-clippy/issues/6546
pub mod issue_6546 {
    type ResInv<A, B> = Result<B, A>;

    pub fn should_lint() -> ResInv<(), usize> {
        //~^ ERROR: this returns a `Result<_, ()>`
        Ok(0)
    }

    pub fn should_not_lint() -> ResInv<usize, ()> {
        Ok(())
    }

    type MyRes<A, B> = Result<(A, B), Box<dyn std::error::Error>>;

    pub fn should_not_lint2(x: i32) -> MyRes<i32, ()> {
        Ok((x, ()))
    }
}

fn main() {}
