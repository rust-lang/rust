static x: impl Fn(&str) -> Result<&str, ()> = move |source| {
    //~^ `impl Trait` not allowed within type [E0562]
    let res = (move |source| Ok(source))(source);
    let res = res.or((move |source| Ok(source))(source));
    res
};

fn main() {}
