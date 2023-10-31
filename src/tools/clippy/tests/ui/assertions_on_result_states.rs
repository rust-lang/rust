//@run-rustfix
#![warn(clippy::assertions_on_result_states)]
#![allow(clippy::unnecessary_literal_unwrap)]

use std::result::Result;

struct Foo;

#[derive(Debug)]
struct DebugFoo;

#[derive(Copy, Clone, Debug)]
struct CopyFoo;

macro_rules! get_ok_macro {
    () => {
        Ok::<_, DebugFoo>(Foo)
    };
}

fn main() {
    // test ok
    let r: Result<Foo, DebugFoo> = Ok(Foo);
    debug_assert!(r.is_ok());
    assert!(r.is_ok());

    // test ok with non-debug error type
    let r: Result<Foo, Foo> = Ok(Foo);
    assert!(r.is_ok());

    // test ok with some messages
    let r: Result<Foo, DebugFoo> = Ok(Foo);
    assert!(r.is_ok(), "oops");

    // test ok with unit error
    let r: Result<Foo, ()> = Ok(Foo);
    assert!(r.is_ok());

    // test temporary ok
    fn get_ok() -> Result<Foo, DebugFoo> {
        Ok(Foo)
    }
    assert!(get_ok().is_ok());

    // test macro ok
    assert!(get_ok_macro!().is_ok());

    // test ok that shouldn't be moved
    let r: Result<CopyFoo, DebugFoo> = Ok(CopyFoo);
    fn test_ref_unmoveable_ok(r: &Result<CopyFoo, DebugFoo>) {
        assert!(r.is_ok());
    }
    test_ref_unmoveable_ok(&r);
    assert!(r.is_ok());
    r.unwrap();

    // test ok that is copied
    let r: Result<CopyFoo, CopyFoo> = Ok(CopyFoo);
    assert!(r.is_ok());
    r.unwrap();

    // test reference to ok
    let r: Result<CopyFoo, CopyFoo> = Ok(CopyFoo);
    fn test_ref_copy_ok(r: &Result<CopyFoo, CopyFoo>) {
        assert!(r.is_ok());
    }
    test_ref_copy_ok(&r);
    r.unwrap();

    // test err
    let r: Result<DebugFoo, Foo> = Err(Foo);
    debug_assert!(r.is_err());
    assert!(r.is_err());

    // test err with non-debug value type
    let r: Result<Foo, Foo> = Err(Foo);
    assert!(r.is_err());
}

#[allow(dead_code)]
fn issue9450() {
    let res: Result<i32, i32> = Ok(1);
    assert!(res.is_err())
}
