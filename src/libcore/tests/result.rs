// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn op1() -> Result<isize, &'static str> { Ok(666) }
fn op2() -> Result<isize, &'static str> { Err("sadface") }

#[test]
fn test_and() {
    assert_eq!(op1().and(Ok(667)).unwrap(), 667);
    assert_eq!(op1().and(Err::<i32, &'static str>("bad")).unwrap_err(),
               "bad");

    assert_eq!(op2().and(Ok(667)).unwrap_err(), "sadface");
    assert_eq!(op2().and(Err::<i32,&'static str>("bad")).unwrap_err(),
               "sadface");
}

#[test]
fn test_and_then() {
    assert_eq!(op1().and_then(|i| Ok::<isize, &'static str>(i + 1)).unwrap(), 667);
    assert_eq!(op1().and_then(|_| Err::<isize, &'static str>("bad")).unwrap_err(),
               "bad");

    assert_eq!(op2().and_then(|i| Ok::<isize, &'static str>(i + 1)).unwrap_err(),
               "sadface");
    assert_eq!(op2().and_then(|_| Err::<isize, &'static str>("bad")).unwrap_err(),
               "sadface");
}

#[test]
fn test_or() {
    assert_eq!(op1().or(Ok::<_, &'static str>(667)).unwrap(), 666);
    assert_eq!(op1().or(Err("bad")).unwrap(), 666);

    assert_eq!(op2().or(Ok::<_, &'static str>(667)).unwrap(), 667);
    assert_eq!(op2().or(Err("bad")).unwrap_err(), "bad");
}

#[test]
fn test_or_else() {
    assert_eq!(op1().or_else(|_| Ok::<isize, &'static str>(667)).unwrap(), 666);
    assert_eq!(op1().or_else(|e| Err::<isize, &'static str>(e)).unwrap(), 666);

    assert_eq!(op2().or_else(|_| Ok::<isize, &'static str>(667)).unwrap(), 667);
    assert_eq!(op2().or_else(|e| Err::<isize, &'static str>(e)).unwrap_err(),
               "sadface");
}

#[test]
fn test_impl_map() {
    assert!(Ok::<isize, isize>(1).map(|x| x + 1) == Ok(2));
    assert!(Err::<isize, isize>(1).map(|x| x + 1) == Err(1));
}

#[test]
fn test_impl_map_err() {
    assert!(Ok::<isize, isize>(1).map_err(|x| x + 1) == Ok(1));
    assert!(Err::<isize, isize>(1).map_err(|x| x + 1) == Err(2));
}

#[test]
fn test_collect() {
    let v: Result<Vec<isize>, ()> = (0..0).map(|_| Ok::<isize, ()>(0)).collect();
    assert!(v == Ok(vec![]));

    let v: Result<Vec<isize>, ()> = (0..3).map(|x| Ok::<isize, ()>(x)).collect();
    assert!(v == Ok(vec![0, 1, 2]));

    let v: Result<Vec<isize>, isize> = (0..3).map(|x| {
        if x > 1 { Err(x) } else { Ok(x) }
    }).collect();
    assert!(v == Err(2));

    // test that it does not take more elements than it needs
    let mut functions: [Box<Fn() -> Result<(), isize>>; 3] =
        [box || Ok(()), box || Err(1), box || panic!()];

    let v: Result<Vec<()>, isize> = functions.iter_mut().map(|f| (*f)()).collect();
    assert!(v == Err(1));
}

#[test]
fn test_fmt_default() {
    let ok: Result<isize, &'static str> = Ok(100);
    let err: Result<isize, &'static str> = Err("Err");

    let s = format!("{:?}", ok);
    assert_eq!(s, "Ok(100)");
    let s = format!("{:?}", err);
    assert_eq!(s, "Err(\"Err\")");
}

#[test]
fn test_unwrap_or() {
    let ok: Result<isize, &'static str> = Ok(100);
    let ok_err: Result<isize, &'static str> = Err("Err");

    assert_eq!(ok.unwrap_or(50), 100);
    assert_eq!(ok_err.unwrap_or(50), 50);
}

#[test]
fn test_unwrap_or_else() {
    fn handler(msg: &'static str) -> isize {
        if msg == "I got this." {
            50
        } else {
            panic!("BadBad")
        }
    }

    let ok: Result<isize, &'static str> = Ok(100);
    let ok_err: Result<isize, &'static str> = Err("I got this.");

    assert_eq!(ok.unwrap_or_else(handler), 100);
    assert_eq!(ok_err.unwrap_or_else(handler), 50);
}

#[test]
#[should_panic]
pub fn test_unwrap_or_else_panic() {
    fn handler(msg: &'static str) -> isize {
        if msg == "I got this." {
            50
        } else {
            panic!("BadBad")
        }
    }

    let bad_err: Result<isize, &'static str> = Err("Unrecoverable mess.");
    let _ : isize = bad_err.unwrap_or_else(handler);
}


#[test]
pub fn test_expect_ok() {
    let ok: Result<isize, &'static str> = Ok(100);
    assert_eq!(ok.expect("Unexpected error"), 100);
}
#[test]
#[should_panic(expected="Got expected error: \"All good\"")]
pub fn test_expect_err() {
    let err: Result<isize, &'static str> = Err("All good");
    err.expect("Got expected error");
}


#[test]
pub fn test_expect_err_err() {
    let ok: Result<&'static str, isize> = Err(100);
    assert_eq!(ok.expect_err("Unexpected ok"), 100);
}
#[test]
#[should_panic(expected="Got expected ok: \"All good\"")]
pub fn test_expect_err_ok() {
    let err: Result<&'static str, isize> = Ok("All good");
    err.expect_err("Got expected ok");
}

#[test]
pub fn test_iter() {
    let ok: Result<isize, &'static str> = Ok(100);
    let mut it = ok.iter();
    assert_eq!(it.size_hint(), (1, Some(1)));
    assert_eq!(it.next(), Some(&100));
    assert_eq!(it.size_hint(), (0, Some(0)));
    assert!(it.next().is_none());
    assert_eq!((&ok).into_iter().next(), Some(&100));

    let err: Result<isize, &'static str> = Err("error");
    assert_eq!(err.iter().next(), None);
}

#[test]
pub fn test_iter_mut() {
    let mut ok: Result<isize, &'static str> = Ok(100);
    for loc in ok.iter_mut() {
        *loc = 200;
    }
    assert_eq!(ok, Ok(200));
    for loc in &mut ok {
        *loc = 300;
    }
    assert_eq!(ok, Ok(300));

    let mut err: Result<isize, &'static str> = Err("error");
    for loc in err.iter_mut() {
        *loc = 200;
    }
    assert_eq!(err, Err("error"));
}

#[test]
pub fn test_unwrap_or_default() {
    assert_eq!(op1().unwrap_or_default(), 666);
    assert_eq!(op2().unwrap_or_default(), 0);
}
