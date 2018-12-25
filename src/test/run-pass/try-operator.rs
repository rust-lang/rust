#![allow(dead_code)]
// ignore-cloudabi no std::fs

use std::fs::File;
use std::io::{Read, self};
use std::num::ParseIntError;
use std::str::FromStr;

fn on_method() -> Result<i32, ParseIntError> {
    Ok("1".parse::<i32>()? + "2".parse::<i32>()?)
}

fn in_chain() -> Result<String, ParseIntError> {
    Ok("3".parse::<i32>()?.to_string())
}

fn on_call() -> Result<i32, ParseIntError> {
    fn parse<T: FromStr>(s: &str) -> Result<T, T::Err> {
        s.parse()
    }

    Ok(parse("4")?)
}

fn nested() -> Result<i32, ParseIntError> {
    Ok("5".parse::<i32>()?.to_string().parse()?)
}

fn on_path() -> Result<i32, ParseIntError> {
    let x = "6".parse::<i32>();

    Ok(x?)
}

fn on_macro() -> Result<i32, ParseIntError> {
    macro_rules! id {
        ($e:expr) => { $e }
    }

    Ok(id!("7".parse::<i32>())?)
}

fn on_parens() -> Result<i32, ParseIntError> {
    let x = "8".parse::<i32>();

    Ok((x)?)
}

fn on_block() -> Result<i32, ParseIntError> {
    let x = "9".parse::<i32>();

    Ok({x}?)
}

fn on_field() -> Result<i32, ParseIntError> {
    struct Pair<A, B> { a: A, b: B }

    let x = Pair { a: "10".parse::<i32>(), b: 0 };

    Ok(x.a?)
}

fn on_tuple_field() -> Result<i32, ParseIntError> {
    let x = ("11".parse::<i32>(), 0);

    Ok(x.0?)
}

fn on_try() -> Result<i32, ParseIntError> {
    let x = "12".parse::<i32>().map(|i| i.to_string().parse::<i32>());

    Ok(x??)
}

fn on_binary_op() -> Result<i32, ParseIntError> {
    let x = 13 - "14".parse::<i32>()?;
    let y = "15".parse::<i32>()? - 16;
    let z = "17".parse::<i32>()? - "18".parse::<i32>()?;

    Ok(x + y + z)
}

fn on_index() -> Result<i32, ParseIntError> {
    let x = [19];
    let y = "0".parse::<usize>();

    Ok(x[y?])
}

fn on_args() -> Result<i32, ParseIntError> {
    fn sub(x: i32, y: i32) -> i32 { x - y }

    let x = "20".parse();
    let y = "21".parse();

    Ok(sub(x?, y?))
}

fn on_if() -> Result<i32, ParseIntError> {
    Ok(if true {
        "22".parse::<i32>()
    } else {
        "23".parse::<i32>()
    }?)
}

fn on_if_let() -> Result<i32, ParseIntError> {
    Ok(if let Ok(..) = "24".parse::<i32>() {
        "25".parse::<i32>()
    } else {
        "26".parse::<i32>()
    }?)
}

fn on_match() -> Result<i32, ParseIntError> {
    Ok(match "27".parse::<i32>() {
        Err(..) => "28".parse::<i32>(),
        Ok(..) => "29".parse::<i32>(),
    }?)
}

fn tight_binding() -> Result<bool, ()> {
    fn ok<T>(x: T) -> Result<T, ()> { Ok(x) }

    let x = ok(true);
    Ok(!x?)
}

// just type check
fn merge_error() -> Result<i32, Error> {
    let mut s = String::new();

    File::open("foo.txt")?.read_to_string(&mut s)?;

    Ok(s.parse::<i32>()? + 1)
}

fn main() {
    assert_eq!(Ok(3), on_method());

    assert_eq!(Ok("3".to_string()), in_chain());

    assert_eq!(Ok(4), on_call());

    assert_eq!(Ok(5), nested());

    assert_eq!(Ok(6), on_path());

    assert_eq!(Ok(7), on_macro());

    assert_eq!(Ok(8), on_parens());

    assert_eq!(Ok(9), on_block());

    assert_eq!(Ok(10), on_field());

    assert_eq!(Ok(11), on_tuple_field());

    assert_eq!(Ok(12), on_try());

    assert_eq!(Ok(-3), on_binary_op());

    assert_eq!(Ok(19), on_index());

    assert_eq!(Ok(-1), on_args());

    assert_eq!(Ok(22), on_if());

    assert_eq!(Ok(25), on_if_let());

    assert_eq!(Ok(29), on_match());

    assert_eq!(Ok(false), tight_binding());
}

enum Error {
    Io(io::Error),
    Parse(ParseIntError),
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Error {
        Error::Io(e)
    }
}

impl From<ParseIntError> for Error {
    fn from(e: ParseIntError) -> Error {
        Error::Parse(e)
    }
}
