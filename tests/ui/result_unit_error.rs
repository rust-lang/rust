#[warn(clippy::result_unit_err)]
#[allow(unused)]

pub fn returns_unit_error() -> Result<u32, ()> {
    Err(())
}

fn private_unit_errors() -> Result<String, ()> {
    Err(())
}

pub trait HasUnitError {
    fn get_that_error(&self) -> Result<bool, ()>;

    fn get_this_one_too(&self) -> Result<bool, ()> {
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
        Ok(0)
    }
}

fn main() {}
