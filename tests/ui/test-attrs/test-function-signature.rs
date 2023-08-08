// compile-flags: --test

#[test]
fn foo() -> Result<(), ()> {
    Ok(())
}

#[test]
fn bar() -> i32 { //~ ERROR the trait bound `i32: Termination` is not satisfied
    0
}

#[test]
fn baz(val: i32) {} //~ ERROR functions used as tests can not have any arguments

#[test]
fn lifetime_generic<'a>() -> Result<(), &'a str> {
    Err("coerce me to any lifetime")
}

#[test]
fn type_generic<T>() {} //~ ERROR functions used as tests can not have any non-lifetime generic parameters

#[test]
fn const_generic<const N: usize>() {} //~ ERROR functions used as tests can not have any non-lifetime generic parameters

// Regression test for <https://github.com/rust-lang/rust/issues/112360>. This used to ICE.
fn nested() {
    #[test]
    fn foo(arg: ()) {} //~ ERROR functions used as tests can not have any arguments
}
