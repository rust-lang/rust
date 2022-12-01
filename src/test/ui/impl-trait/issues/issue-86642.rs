static x: impl Fn(&str) -> Result<&str, ()> = move |source| {
    //~^ `impl Trait` isn't allowed within type [E0562]
    let res = (move |source| Ok(source))(source);
    let res = res.or((move |source| Ok(source))(source));
    res
};

fn main() {}
