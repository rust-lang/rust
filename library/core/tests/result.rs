use core::ops::DerefMut;
use core::option::*;

fn op1() -> Result<isize, &'static str> {
    Ok(666)
}
fn op2() -> Result<isize, &'static str> {
    Err("sadface")
}

#[test]
fn test_and() {
    assert_eq!(op1().and(Ok(667)).unwrap(), 667);
    assert_eq!(op1().and(Err::<i32, &'static str>("bad")).unwrap_err(), "bad");

    assert_eq!(op2().and(Ok(667)).unwrap_err(), "sadface");
    assert_eq!(op2().and(Err::<i32, &'static str>("bad")).unwrap_err(), "sadface");
}

#[test]
fn test_and_then() {
    assert_eq!(op1().and_then(|i| Ok::<isize, &'static str>(i + 1)).unwrap(), 667);
    assert_eq!(op1().and_then(|_| Err::<isize, &'static str>("bad")).unwrap_err(), "bad");

    assert_eq!(op2().and_then(|i| Ok::<isize, &'static str>(i + 1)).unwrap_err(), "sadface");
    assert_eq!(op2().and_then(|_| Err::<isize, &'static str>("bad")).unwrap_err(), "sadface");
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
    assert_eq!(op2().or_else(|e| Err::<isize, &'static str>(e)).unwrap_err(), "sadface");
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

    let v: Result<Vec<isize>, isize> = (0..3).map(|x| if x > 1 { Err(x) } else { Ok(x) }).collect();
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
fn test_ok_or_err() {
    let ok: Result<isize, isize> = Ok(100);
    let err: Result<isize, isize> = Err(200);

    assert_eq!(ok.into_ok_or_err(), 100);
    assert_eq!(err.into_ok_or_err(), 200);
}

#[test]
fn test_unwrap_or_else() {
    fn handler(msg: &'static str) -> isize {
        if msg == "I got this." { 50 } else { panic!("BadBad") }
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
        if msg == "I got this." { 50 } else { panic!("BadBad") }
    }

    let bad_err: Result<isize, &'static str> = Err("Unrecoverable mess.");
    let _: isize = bad_err.unwrap_or_else(handler);
}

#[test]
fn test_unwrap_unchecked() {
    let ok: Result<isize, &'static str> = Ok(100);
    assert_eq!(unsafe { ok.unwrap_unchecked() }, 100);
}

#[test]
fn test_unwrap_err_unchecked() {
    let ok_err: Result<isize, &'static str> = Err("Err");
    assert_eq!(unsafe { ok_err.unwrap_err_unchecked() }, "Err");
}

#[test]
pub fn test_expect_ok() {
    let ok: Result<isize, &'static str> = Ok(100);
    assert_eq!(ok.expect("Unexpected error"), 100);
}
#[test]
#[should_panic(expected = "Got expected error: \"All good\"")]
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
#[should_panic(expected = "Got expected ok: \"All good\"")]
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
pub fn test_into_ok() {
    fn infallible_op() -> Result<isize, !> {
        Ok(666)
    }

    assert_eq!(infallible_op().into_ok(), 666);

    enum MyNeverToken {}
    impl From<MyNeverToken> for ! {
        fn from(never: MyNeverToken) -> ! {
            match never {}
        }
    }

    fn infallible_op2() -> Result<isize, MyNeverToken> {
        Ok(667)
    }

    assert_eq!(infallible_op2().into_ok(), 667);
}

#[test]
pub fn test_into_err() {
    fn until_error_op() -> Result<!, isize> {
        Err(666)
    }

    assert_eq!(until_error_op().into_err(), 666);

    enum MyNeverToken {}
    impl From<MyNeverToken> for ! {
        fn from(never: MyNeverToken) -> ! {
            match never {}
        }
    }

    fn until_error_op2() -> Result<MyNeverToken, isize> {
        Err(667)
    }

    assert_eq!(until_error_op2().into_err(), 667);
}

#[test]
fn test_try() {
    fn try_result_ok() -> Result<u8, u32> {
        let result: Result<u8, u8> = Ok(1);
        let val = result?;
        Ok(val)
    }
    assert_eq!(try_result_ok(), Ok(1));

    fn try_result_err() -> Result<u8, u32> {
        let result: Result<u8, u8> = Err(1);
        let val = result?;
        Ok(val)
    }
    assert_eq!(try_result_err(), Err(1));
}

#[test]
fn test_result_as_deref() {
    // &Result<T: Deref, E>::Ok(T).as_deref() ->
    //      Result<&T::Deref::Target, &E>::Ok(&*T)
    let ref_ok = &Result::Ok::<&i32, u8>(&42);
    let expected_result = Result::Ok::<&i32, &u8>(&42);
    assert_eq!(ref_ok.as_deref(), expected_result);

    let ref_ok = &Result::Ok::<String, u32>(String::from("a result"));
    let expected_result = Result::Ok::<&str, &u32>("a result");
    assert_eq!(ref_ok.as_deref(), expected_result);

    let ref_ok = &Result::Ok::<Vec<i32>, u32>(vec![1, 2, 3, 4, 5]);
    let expected_result = Result::Ok::<&[i32], &u32>([1, 2, 3, 4, 5].as_slice());
    assert_eq!(ref_ok.as_deref(), expected_result);

    // &Result<T: Deref, E>::Err(T).as_deref() ->
    //      Result<&T::Deref::Target, &E>::Err(&*E)
    let val = 41;
    let ref_err = &Result::Err::<&u8, i32>(val);
    let expected_result = Result::Err::<&u8, &i32>(&val);
    assert_eq!(ref_err.as_deref(), expected_result);

    let s = String::from("an error");
    let ref_err = &Result::Err::<&u32, String>(s.clone());
    let expected_result = Result::Err::<&u32, &String>(&s);
    assert_eq!(ref_err.as_deref(), expected_result);

    let v = vec![5, 4, 3, 2, 1];
    let ref_err = &Result::Err::<&u32, Vec<i32>>(v.clone());
    let expected_result = Result::Err::<&u32, &Vec<i32>>(&v);
    assert_eq!(ref_err.as_deref(), expected_result);
}

#[test]
fn test_result_as_deref_mut() {
    // &mut Result<T: DerefMut, E>::Ok(T).as_deref_mut() ->
    //      Result<&mut T::DerefMut::Target, &mut E>::Ok(&mut *T)
    let mut val = 42;
    let mut expected_val = 42;
    let mut_ok = &mut Result::Ok::<&mut i32, u8>(&mut val);
    let expected_result = Result::Ok::<&mut i32, &mut u8>(&mut expected_val);
    assert_eq!(mut_ok.as_deref_mut(), expected_result);

    let mut expected_string = String::from("a result");
    let mut_ok = &mut Result::Ok::<String, u32>(expected_string.clone());
    let expected_result = Result::Ok::<&mut str, &mut u32>(expected_string.deref_mut());
    assert_eq!(mut_ok.as_deref_mut(), expected_result);

    let mut expected_vec = vec![1, 2, 3, 4, 5];
    let mut_ok = &mut Result::Ok::<Vec<i32>, u32>(expected_vec.clone());
    let expected_result = Result::Ok::<&mut [i32], &mut u32>(expected_vec.as_mut_slice());
    assert_eq!(mut_ok.as_deref_mut(), expected_result);

    // &mut Result<T: DerefMut, E>::Err(T).as_deref_mut() ->
    //      Result<&mut T, &mut E>::Err(&mut *E)
    let mut val = 41;
    let mut_err = &mut Result::Err::<&mut u8, i32>(val);
    let expected_result = Result::Err::<&mut u8, &mut i32>(&mut val);
    assert_eq!(mut_err.as_deref_mut(), expected_result);

    let mut expected_string = String::from("an error");
    let mut_err = &mut Result::Err::<&mut u32, String>(expected_string.clone());
    let expected_result = Result::Err::<&mut u32, &mut String>(&mut expected_string);
    assert_eq!(mut_err.as_deref_mut(), expected_result);

    let mut expected_vec = vec![5, 4, 3, 2, 1];
    let mut_err = &mut Result::Err::<&mut u32, Vec<i32>>(expected_vec.clone());
    let expected_result = Result::Err::<&mut u32, &mut Vec<i32>>(&mut expected_vec);
    assert_eq!(mut_err.as_deref_mut(), expected_result);
}

#[test]
fn result_const() {
    // test that the methods of `Result` are usable in a const context

    const RESULT: Result<usize, bool> = Ok(32);

    const REF: Result<&usize, &bool> = RESULT.as_ref();
    assert_eq!(REF, Ok(&32));

    const IS_OK: bool = RESULT.is_ok();
    assert!(IS_OK);

    const IS_ERR: bool = RESULT.is_err();
    assert!(!IS_ERR)
}

#[test]
const fn result_const_mut() {
    let mut result: Result<usize, bool> = Ok(32);

    {
        let as_mut = result.as_mut();
        match as_mut {
            Ok(v) => *v = 42,
            Err(_) => unreachable!(),
        }
    }

    let mut result_err: Result<usize, bool> = Err(false);

    {
        let as_mut = result_err.as_mut();
        match as_mut {
            Ok(_) => unreachable!(),
            Err(v) => *v = true,
        }
    }
}

#[test]
fn result_opt_conversions() {
    #[derive(Copy, Clone, Debug, PartialEq)]
    struct BadNumErr;

    fn try_num(x: i32) -> Result<i32, BadNumErr> {
        if x <= 5 { Ok(x + 1) } else { Err(BadNumErr) }
    }

    type ResOpt = Result<Option<i32>, BadNumErr>;
    type OptRes = Option<Result<i32, BadNumErr>>;

    let mut x: ResOpt = Ok(Some(5));
    let mut y: OptRes = Some(Ok(5));
    assert_eq!(x, y.transpose());
    assert_eq!(x.transpose(), y);

    x = Ok(None);
    y = None;
    assert_eq!(x, y.transpose());
    assert_eq!(x.transpose(), y);

    x = Err(BadNumErr);
    y = Some(Err(BadNumErr));
    assert_eq!(x, y.transpose());
    assert_eq!(x.transpose(), y);

    let res: Result<Vec<i32>, BadNumErr> = (0..10)
        .map(|x| {
            let y = try_num(x)?;
            Ok(if y % 2 == 0 { Some(y - 1) } else { None })
        })
        .filter_map(Result::transpose)
        .collect();

    assert_eq!(res, Err(BadNumErr))
}

#[test]
fn result_try_trait_v2_branch() {
    use core::num::NonZeroU32;
    use core::ops::{ControlFlow::*, Try};
    assert_eq!(Ok::<i32, i32>(4).branch(), Continue(4));
    assert_eq!(Err::<i32, i32>(4).branch(), Break(Err(4)));
    let one = NonZeroU32::new(1).unwrap();
    assert_eq!(Ok::<(), NonZeroU32>(()).branch(), Continue(()));
    assert_eq!(Err::<(), NonZeroU32>(one).branch(), Break(Err(one)));
    assert_eq!(Ok::<NonZeroU32, ()>(one).branch(), Continue(one));
    assert_eq!(Err::<NonZeroU32, ()>(()).branch(), Break(Err(())));
}
