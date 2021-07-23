static x: impl Fn(&str) -> Result<&str, ()> = move |source| {
    //~^ `impl Trait` not allowed outside of function and method return types
    let res = (move |source| Ok(source))(source);
    let res = res.or((move |source| Ok(source))(source));
    res
};

fn main() {}
