//@aux-build:option_helpers.rs
#![warn(clippy::manual_is_variant_and)]

#[macro_use]
extern crate option_helpers;

#[rustfmt::skip]
fn option_methods() {
    let opt = Some(1);

    // Check for `option.map(_).unwrap_or_default()` use.
    // Single line case.
    let _ = opt.map(|x| x > 1)
    //~^ manual_is_variant_and
        // Should lint even though this call is on a separate line.
        .unwrap_or_default();
    // Multi-line cases.
    let _ = opt.map(|x| {
    //~^ manual_is_variant_and
        x > 1
    }
    ).unwrap_or_default();
    let _ = opt.map(|x| x > 1).unwrap_or_default();
    //~^ manual_is_variant_and
    let _ = opt
        .map(|x| x > 1)
        //~^ manual_is_variant_and
        .unwrap_or_default();

    // won't fix because the return type of the closure is not `bool`
    let _ = opt.map(|x| x + 1).unwrap_or_default();

    let opt2 = Some('a');
    let _ = opt2.map(char::is_alphanumeric).unwrap_or_default(); // should lint
    //~^ manual_is_variant_and
    let _ = opt_map!(opt2, |x| x == 'a').unwrap_or_default(); // should not lint
}

#[rustfmt::skip]
fn result_methods() {
    let res: Result<i32, ()> = Ok(1);

    // multi line cases
    let _ = res.map(|x| {
    //~^ manual_is_variant_and
        x > 1
    }
    ).unwrap_or_default();
    let _ = res.map(|x| x > 1)
    //~^ manual_is_variant_and
        .unwrap_or_default();

    // won't fix because the return type of the closure is not `bool`
    let _ = res.map(|x| x + 1).unwrap_or_default();

    let res2: Result<char, ()> = Ok('a');
    let _ = res2.map(char::is_alphanumeric).unwrap_or_default(); // should lint
    //~^ manual_is_variant_and
    let _ = opt_map!(res2, |x| x == 'a').unwrap_or_default(); // should not lint
}

fn main() {
    option_methods();
    result_methods();
}
