#![warn(clippy::option_map_unit_fn)]
#![allow(unused)]

fn do_nothing<T>(_: T) {}

fn diverge<T>(_: T) -> ! {
    panic!()
}

fn plus_one(value: usize) -> usize {
    value + 1
}

#[rustfmt::skip]
fn option_map_unit_fn() {

    x.field.map(|value| { do_nothing(value); do_nothing(value) });
    //~^ ERROR: cannot find value

    x.field.map(|value| if value > 0 { do_nothing(value); do_nothing(value) });
    //~^ ERROR: cannot find value

    // Suggestion for the let block should be `{ ... }` as it's too difficult to build a
    // proper suggestion for these cases
    x.field.map(|value| {
        do_nothing(value);
        do_nothing(value)
    });
    //~^^^^ ERROR: cannot find value
    x.field.map(|value| { do_nothing(value); do_nothing(value); });
    //~^ ERROR: cannot find value

    // The following should suggest `if let Some(_X) ...` as it's difficult to generate a proper let variable name for them
    Some(42).map(diverge);
    "12".parse::<i32>().ok().map(diverge);
    Some(plus_one(1)).map(do_nothing);

    // Should suggest `if let Some(_y) ...` to not override the existing foo variable
    let y = Some(42);
    y.map(do_nothing);
}

fn main() {}
