#![allow(unused_assignments)]
#![cfg_attr(rustfmt, rustfmt::skip)]
//@ failure-status: 1

fn call(return_error: bool) -> Result<(), ()> {
    if return_error {
        Err(())
    } else {
        Ok(())
    }
}

fn test1() -> Result<(), ()> {
    let mut
        countdown = 10
    ;
    for
        _
    in
        0..10
    {
        countdown
            -= 1
        ;
        if
            countdown < 5
        {
            call(/*return_error=*/ true)?;
            call(/*return_error=*/ false)?;
        }
        else
        {
            call(/*return_error=*/ false)?;
        }
    }
    Ok(())
}

struct Thing1;
impl Thing1 {
    fn get_thing_2(&self, return_error: bool) -> Result<Thing2, ()> {
        if return_error {
            Err(())
        } else {
            Ok(Thing2 {})
        }
    }
}

struct Thing2;
impl Thing2 {
    fn call(&self, return_error: bool) -> Result<u32, ()> {
        if return_error {
            Err(())
        } else {
            Ok(57)
        }
    }
}

fn test2() -> Result<(), ()> {
    let thing1 = Thing1{};
    let mut
        countdown = 10
    ;
    for
        _
    in
        0..10
    {
        countdown
            -= 1
        ;
        if
            countdown < 5
        {
            thing1.get_thing_2(/*err=*/ false)?.call(/*err=*/ true).expect_err("call should fail");
            thing1
                .
                get_thing_2(/*return_error=*/ false)
                ?
                .
                call(/*return_error=*/ true)
                .
                expect_err(
                    "call should fail"
                );
            let val = thing1.get_thing_2(/*return_error=*/ true)?.call(/*return_error=*/ true)?;
            assert_eq!(val, 57);
            let val = thing1.get_thing_2(/*return_error=*/ true)?.call(/*return_error=*/ false)?;
            assert_eq!(val, 57);
        }
        else
        {
            let val = thing1.get_thing_2(/*return_error=*/ false)?.call(/*return_error=*/ false)?;
            assert_eq!(val, 57);
            let val = thing1
                .get_thing_2(/*return_error=*/ false)?
                .call(/*return_error=*/ false)?;
            assert_eq!(val, 57);
            let val = thing1
                .get_thing_2(/*return_error=*/ false)
                ?
                .call(/*return_error=*/ false)
                ?
                ;
            assert_eq!(val, 57);
        }
    }
    Ok(())
}

fn main() -> Result<(), ()> {
    test1().expect_err("test1 should fail");
    test2()
    ?
    ;
    Ok(())
}
