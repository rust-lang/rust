static x: impl Fn(&str) -> Result<&str, ()> = move |source| {
    //~^ ERROR `impl Trait` is not allowed in static types
    //~| ERROR cycle detected when computing type of `x`
    //~| ERROR the placeholder `_` is not allowed within types on item signatures for static variables
    let res = (move |source| Ok(source))(source);
    let res = res.or((move |source| Ok(source))(source));
    res
};

fn main() {}
