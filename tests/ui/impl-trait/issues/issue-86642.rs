static x: impl Fn(&str) -> Result<&str, ()> = move |source| {
    //~^ ERROR `impl Trait` is not allowed in static types
    let res = (move |source| Ok(source))(source);
    let res = res.or((move |source| Ok(source))(source));
    res
};

fn main() {}
