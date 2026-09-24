fn foo() -> Option<usize> {
    let _ = Some(42).expect("moop")?;
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    let _ = Some(42).unwrap()?;
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    None
}

fn bar() -> Option<usize> {
    let _ = foo().or(Some(43)).unwrap()?;
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    None
}

fn baz() -> Result<usize, ()> {
    let _ = Ok(44).unwrap()?;
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    Ok(0)
}

struct FakeUnwrappable;

impl FakeUnwrappable {
    fn unwrap(self) -> () {}
}

fn qux() -> Option<usize> {
    FakeUnwrappable.unwrap()?;
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    None
}

fn main() {}
