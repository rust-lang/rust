struct MyError;

fn foo(x: bool) -> Result<(), MyError> {
    if x {
        Err(MyError);
        //~^ ERROR type annotations needed
    }

    Ok(())
}

fn bar(x: bool) -> Result<(), MyError> {
    if x {
        Ok(());
        //~^ ERROR type annotations needed
    }

    Ok(())
}

fn baz(x: bool) -> Result<(), MyError> {
    //~^ ERROR mismatched types
    if x {
        1;
    }

    Err(MyError);
}

fn error() -> Result<(), MyError> {
    Err(MyError)
}

fn bak(x: bool) -> Result<(), MyError> {
    if x {
        //~^ ERROR mismatched types
        error();
    } else {
        //~^ ERROR mismatched types
        error();
    }
}

fn bad(x: bool) -> Result<(), MyError> {
    Err(MyError); //~ ERROR type annotations needed
    Ok(())
}

fn with_closure<F, A, B>(_: F) -> i32
where
    F: FnOnce(A, B),
{
    0
}

fn a() -> i32 {
    with_closure(|x: u32, y| {}); //~ ERROR type annotations needed
    0
}

fn main() {}
