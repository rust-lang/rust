// Ensure we don't suggest tuple-wrapping when we'd end up with a type error

fn main() {
    // we shouldn't suggest to fix these - `2` isn't a `bool`

    let _: Option<(i32, bool)> = Some(1, 2);
    //~^ ERROR this enum variant takes 1 argument but 2 arguments were supplied
    int_bool(1, 2);
    //~^ ERROR this function takes 1 argument but 2 arguments were supplied

    let _: Option<(i8,)> = Some();
    //~^ ERROR this enum variant takes 1 argument but 0 arguments were supplied

    let _: Option<(i32,)> = Some(5_usize);
    //~^ ERROR mismatched types

    let _: Option<(i32,)> = Some((5_usize));
    //~^ ERROR mismatched types
}

fn int_bool(_: (i32, bool)) {
}
