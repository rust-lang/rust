#![allow(unused)]
#![warn(clippy::unnecessary_map_on_constructor)]

use std::ffi::OsStr;

fn fun(t: i32) -> i32 {
    t
}

fn notfun(e: SimpleError) -> SimpleError {
    e
}
macro_rules! expands_to_fun {
    () => {
        fun
    };
}

#[derive(Copy, Clone)]
struct SimpleError {}

type SimpleResult = std::result::Result<i32, SimpleError>;

fn main() {
    let x: i32 = 4;

    let err = SimpleError {};
    let a = Some(x);
    let b: SimpleResult = Ok(x);
    let c: SimpleResult = Err(err);

    let a = Some(x).map(fun);
    //~^ unnecessary_map_on_constructor
    let b: SimpleResult = Ok(x).map(fun);
    //~^ unnecessary_map_on_constructor
    let c: SimpleResult = Err(err).map_err(notfun);
    //~^ unnecessary_map_on_constructor

    let a = Option::Some(x).map(fun);
    //~^ unnecessary_map_on_constructor
    let b: SimpleResult = SimpleResult::Ok(x).map(fun);
    //~^ unnecessary_map_on_constructor
    let c: SimpleResult = SimpleResult::Err(err).map_err(notfun);
    //~^ unnecessary_map_on_constructor
    let b: std::result::Result<i32, SimpleError> = Ok(x).map(fun);
    //~^ unnecessary_map_on_constructor
    let c: std::result::Result<i32, SimpleError> = Err(err).map_err(notfun);
    //~^ unnecessary_map_on_constructor

    let a = Some(fun(x));
    let b: SimpleResult = Ok(fun(x));
    let c: SimpleResult = Err(notfun(err));

    // Should not trigger warning
    a.map(fun);
    b.map(fun);
    c.map_err(notfun);

    b.map_err(notfun); // Ok(_).map_err
    c.map(fun); // Err(_).map()

    option_env!("PATH").map(OsStr::new);
    Some(x).map(expands_to_fun!());
}
