#![warn(clippy::manual_ok_or)]
#![allow(clippy::or_fun_call)]
#![allow(clippy::disallowed_names)]
#![allow(clippy::redundant_closure)]
#![allow(dead_code)]
#![allow(unused_must_use)]

fn main() {
    // basic case
    let foo: Option<i32> = None;
    foo.map_or(Err("error"), |v| Ok(v));
    //~^ manual_ok_or

    // eta expansion case
    foo.map_or(Err("error"), Ok);
    //~^ manual_ok_or

    // turbo fish syntax
    None::<i32>.map_or(Err("error"), |v| Ok(v));
    //~^ manual_ok_or

    // multiline case
    #[rustfmt::skip]
    foo.map_or(Err::<i32, &str>(
    //~^ manual_ok_or
        &format!(
            "{}{}{}{}{}{}{}",
            "Alice", "Bob", "Sarah", "Marc", "Sandra", "Eric", "Jenifer")
        ),
        |v| Ok(v),
    );

    // not applicable, closure isn't direct `Ok` wrapping
    foo.map_or(Err("error"), |v| Ok(v + 1));

    // not applicable, or side isn't `Result::Err`
    foo.map_or(Ok::<i32, &str>(1), |v| Ok(v));

    // not applicable, expr is not a `Result` value
    foo.map_or(42, |v| v);

    // TODO patterns not covered yet
    match foo {
        Some(v) => Ok(v),
        None => Err("error"),
    };
    foo.map_or_else(|| Err("error"), |v| Ok(v));
}
