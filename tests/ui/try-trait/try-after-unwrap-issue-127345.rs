fn foo() -> Option<usize> {
    let x = Some(42).expect("moop")?;
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| HELP remove the `.expect()`
    let x = Some(42).unwrap()?;
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| HELP remove the `.unwrap()`
    x
}

fn bar() -> Option<usize> {
    foo().or(Some(43)).unwrap()?
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    //~| HELP the trait `Try` is not implemented for `usize`
    //~| HELP remove the `.unwrap()`
}

fn baz() -> Result<usize, ()> {
    Ok(44).unwrap()?
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| HELP remove the `.unwrap()`
}

fn baz2() -> Result<String, ()> {
    Ok(44).unwrap()?
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| HELP remove the `.unwrap()`
}

fn baz3() -> Option<usize> {
    Ok(44).unwrap()?
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| HELP remove the `.unwrap()`
}

fn baz4() {
    Ok(44).unwrap()?
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| HELP remove the `.unwrap()`
    //~| ERROR the `?` operator can only be used in a function that returns `Result` or `Option` (or another type that implements `FromResidual`)
    //~| HELP the trait `FromResidual<_>` is not implemented for `()`
}

struct FakeUnwrappable;

impl FakeUnwrappable {
    fn unwrap(self) -> () {}
}

fn qux() -> Option<usize> {
    FakeUnwrappable.unwrap()?
    //~^ ERROR the `?` operator can only be applied to values that implement `Try`
    //~| HELP the trait `Try` is not implemented for `()`
}

fn main() {}
