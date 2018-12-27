// run-pass
use std::num::{ParseFloatError, ParseIntError};

fn main() {
    assert_eq!(simple(), Ok(1));
    assert_eq!(nested(), Ok(2));
    assert_eq!(merge_ok(), Ok(3.0));
    assert_eq!(merge_int_err(), Err(Error::Int));
    assert_eq!(merge_float_err(), Err(Error::Float));
}

fn simple() -> Result<i32, ParseIntError> {
    Ok(try!("1".parse()))
}

fn nested() -> Result<i32, ParseIntError> {
    Ok(try!(try!("2".parse::<i32>()).to_string().parse::<i32>()))
}

fn merge_ok() -> Result<f32, Error> {
    Ok(try!("1".parse::<i32>()) as f32 + try!("2.0".parse::<f32>()))
}

fn merge_int_err() -> Result<f32, Error> {
    Ok(try!("a".parse::<i32>()) as f32 + try!("2.0".parse::<f32>()))
}

fn merge_float_err() -> Result<f32, Error> {
    Ok(try!("1".parse::<i32>()) as f32 + try!("b".parse::<f32>()))
}

#[derive(Debug, PartialEq)]
enum Error {
    Int,
    Float,
}

impl From<ParseIntError> for Error {
    fn from(_: ParseIntError) -> Error {
        Error::Int
    }
}

impl From<ParseFloatError> for Error {
    fn from(_: ParseFloatError) -> Error {
        Error::Float
    }
}
