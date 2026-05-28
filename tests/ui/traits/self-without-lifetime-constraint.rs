use std::error::Error;
use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ValueRef<'a> {
    Null,
    Integer(i64),
    Real(f64),
    Text(&'a [u8]),
    Blob(&'a [u8]),
}

impl<'a> ValueRef<'a> {
    pub fn as_str(&self) -> FromSqlResult<&'a str, &'a &'a str> {
        match *self {
            ValueRef::Text(t) => {
                std::str::from_utf8(t).map_err(|_| FromSqlError::InvalidType).map(|x| (x, &x))
                //~^ ERROR: cannot return value referencing function parameter `x`
            }
            _ => Err(FromSqlError::InvalidType),
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum FromSqlError {
    InvalidType
}

impl fmt::Display for FromSqlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InvalidType")
    }
}

impl Error for FromSqlError {}

pub type FromSqlResult<T, K> = Result<(T, K), FromSqlError>;

pub trait FromSql: Sized {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self, &Self>;
}

impl FromSql for &str {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<&str, &&str> {
    //~^ ERROR `impl` item signature doesn't match `trait` item signature
        value.as_str()
    }
}

pub fn main() {
    println!("{}", "Hello World");
}
