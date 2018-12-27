use core::option::*;

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
    let mut functions: [Box<dyn Fn() -> Result<(), isize>>; 3] =
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

#[test]
fn test_try() {
    fn try_result_some() -> Option<u8> {
        let val = Ok(1)?;
        Some(val)
    }
    assert_eq!(try_result_some(), Some(1));

    fn try_result_none() -> Option<u8> {
        let val = Err(NoneError)?;
        Some(val)
    }
    assert_eq!(try_result_none(), None);

    fn try_result_ok() -> Result<u8, u8> {
        let result: Result<u8, u8> = Ok(1);
        let val = result?;
        Ok(val)
    }
    assert_eq!(try_result_ok(), Ok(1));

    fn try_result_err() -> Result<u8, u8> {
        let result: Result<u8, u8> = Err(1);
        let val = result?;
        Ok(val)
    }
    assert_eq!(try_result_err(), Err(1));
}

#[test]
fn test_result_deref() {
    // &Result<T: Deref, E>::Ok(T).deref_ok() ->
    //      Result<&T::Deref::Target, &E>::Ok(&*T)
    let ref_ok = &Result::Ok::<&i32, u8>(&42);
    let expected_result = Result::Ok::<&i32, &u8>(&42);
    assert_eq!(ref_ok.deref_ok(), expected_result);

    let ref_ok = &Result::Ok::<String, u32>(String::from("a result"));
    let expected_result = Result::Ok::<&str, &u32>("a result");
    assert_eq!(ref_ok.deref_ok(), expected_result);

    let ref_ok = &Result::Ok::<Vec<i32>, u32>(vec![1, 2, 3, 4, 5]);
    let expected_result = Result::Ok::<&[i32], &u32>(&[1, 2, 3, 4, 5][..]);
    assert_eq!(ref_ok.deref_ok(), expected_result);

    // &Result<T: Deref, E: Deref>::Ok(T).deref() ->
    //      Result<&T::Deref::Target, &E::Deref::Target>::Ok(&*T)
    let ref_ok = &Result::Ok::<&i32, &u8>(&42);
    let expected_result = Result::Ok::<&i32, &u8>(&42);
    assert_eq!(ref_ok.deref(), expected_result);

    let ref_ok = &Result::Ok::<String, &u32>(String::from("a result"));
    let expected_result = Result::Ok::<&str, &u32>("a result");
    assert_eq!(ref_ok.deref(), expected_result);

    let ref_ok = &Result::Ok::<Vec<i32>, &u32>(vec![1, 2, 3, 4, 5]);
    let expected_result = Result::Ok::<&[i32], &u32>(&[1, 2, 3, 4, 5][..]);
    assert_eq!(ref_ok.deref(), expected_result);

    // &Result<T, E: Deref>::Err(T).deref_err() ->
    //      Result<&T, &E::Deref::Target>::Err(&*E)
    let ref_err = &Result::Err::<u8, &i32>(&41);
    let expected_result = Result::Err::<&u8, &i32>(&41);
    assert_eq!(ref_err.deref_err(), expected_result);

    let ref_err = &Result::Err::<u32, String>(String::from("an error"));
    let expected_result = Result::Err::<&u32, &str>("an error");
    assert_eq!(ref_err.deref_err(), expected_result);

    let ref_err = &Result::Err::<u32, Vec<i32>>(vec![5, 4, 3, 2, 1]);
    let expected_result = Result::Err::<&u32, &[i32]>(&[5, 4, 3, 2, 1][..]);
    assert_eq!(ref_err.deref_err(), expected_result);

    // &Result<T: Deref, E: Deref>::Err(T).deref_err() ->
    //      Result<&T, &E::Deref::Target>::Err(&*E)
    let ref_err = &Result::Err::<&u8, &i32>(&41);
    let expected_result = Result::Err::<&u8, &i32>(&41);
    assert_eq!(ref_err.deref(), expected_result);

    let ref_err = &Result::Err::<&u32, String>(String::from("an error"));
    let expected_result = Result::Err::<&u32, &str>("an error");
    assert_eq!(ref_err.deref(), expected_result);

    let ref_err = &Result::Err::<&u32, Vec<i32>>(vec![5, 4, 3, 2, 1]);
    let expected_result = Result::Err::<&u32, &[i32]>(&[5, 4, 3, 2, 1][..]);
    assert_eq!(ref_err.deref(), expected_result);

    // The following cases test calling deref_* with the wrong variant (i.e.
    // `deref_ok()` with a `Result::Err()`, or `deref_err()` with a `Result::Ok()`.
    // While unusual, these cases are supported to ensure that an `inner_deref`
    // call can still be made even when one of the Result types does not implement
    // `Deref` (for example, std::io::Error).

    // &Result<T, E: Deref>::Ok(T).deref_err() ->
    //      Result<&T, &E::Deref::Target>::Ok(&T)
    let ref_ok = &Result::Ok::<i32, &u8>(42);
    let expected_result = Result::Ok::<&i32, &u8>(&42);
    assert_eq!(ref_ok.deref_err(), expected_result);

    let ref_ok = &Result::Ok::<&str, &u32>("a result");
    let expected_result = Result::Ok::<&&str, &u32>(&"a result");
    assert_eq!(ref_ok.deref_err(), expected_result);

    let ref_ok = &Result::Ok::<[i32; 5], &u32>([1, 2, 3, 4, 5]);
    let expected_result = Result::Ok::<&[i32; 5], &u32>(&[1, 2, 3, 4, 5]);
    assert_eq!(ref_ok.deref_err(), expected_result);

    // &Result<T: Deref, E>::Err(E).deref_ok() ->
    //      Result<&T::Deref::Target, &E>::Err(&E)
    let ref_err = &Result::Err::<&u8, i32>(41);
    let expected_result = Result::Err::<&u8, &i32>(&41);
    assert_eq!(ref_err.deref_ok(), expected_result);

    let ref_err = &Result::Err::<&u32, &str>("an error");
    let expected_result = Result::Err::<&u32, &&str>(&"an error");
    assert_eq!(ref_err.deref_ok(), expected_result);

    let ref_err = &Result::Err::<&u32, [i32; 5]>([5, 4, 3, 2, 1]);
    let expected_result = Result::Err::<&u32, &[i32; 5]>(&[5, 4, 3, 2, 1]);
    assert_eq!(ref_err.deref_ok(), expected_result);
}
