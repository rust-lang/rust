// edition:2021

fn foo(b: bool) -> impl std::fmt::Debug {
    if b {
        return vec![42]
    }
    [].into_iter().collect()
}

fn bar(b: bool) -> impl std::fmt::Debug {
    if b {
        return [].into_iter().collect()
    }
    vec![42]
}

fn bak(b: bool) -> impl std::fmt::Debug {
    if b {
        return std::iter::empty().collect()
    }
    vec![42]
}

fn baa(b: bool) -> impl std::fmt::Debug {
    if b {
        return [42].into_iter().collect()
    }
    vec![]
}

fn muh() -> Result<(), impl std::fmt::Debug> {
    Err("whoops")?; //~ ERROR `?` couldn't convert the error to `impl Debug`
    Ok(())
}

fn muh2() -> Result<(), impl std::fmt::Debug> {
    return Err(From::from("foo")); //~ ERROR the trait bound `impl Debug: From<&str>` is not satisfied
    Ok(())
}

fn muh3() -> Result<(), impl std::fmt::Debug> {
    Err(From::from("foo")) //~ ERROR the trait bound `impl Debug: From<&str>` is not satisfied
}

fn main() {}
