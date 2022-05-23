static x: impl Fn(&str) -> Result<&str, ()> = move |source| {
    //~^ `impl Trait` only allowed in function and inherent method return types
    let res = (move |source| Ok(source))(source);
    let res = res.or((move |source| Ok(source))(source));
    res
};

fn main() {}
