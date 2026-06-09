//@no-rustfix
#![warn(clippy::option_map_unit_fn)]
#![allow(clippy::unnecessary_wraps, clippy::unnecessary_map_on_constructor)] // only fires before the fix

fn do_nothing<T>(_: T) {}

fn diverge<T>(_: T) -> ! {
    panic!()
}

fn plus_one(value: usize) -> usize {
    value + 1
}

struct HasOption {
    field: Option<usize>,
}

#[rustfmt::skip]
fn option_map_unit_fn() {
    let x = HasOption { field: Some(10) };

    x.field.map(|value| { do_nothing(value); do_nothing(value) });
    //~^ option_map_unit_fn

    x.field.map(|value| if value > 0 { do_nothing(value); do_nothing(value) });
    //~^ option_map_unit_fn

    // Suggestion for the let block should be `{ ... }` as it's too difficult to build a
    // proper suggestion for these cases
    x.field.map(|value| {
        do_nothing(value);
        do_nothing(value)
    });
    //~^^^^ option_map_unit_fn
    x.field.map(|value| { do_nothing(value); do_nothing(value); });
    //~^ option_map_unit_fn

    // The following should suggest `if let Some(_X) ...` as it's difficult to generate a proper let variable name for them
    Some(42).map(diverge);
    //~^ option_map_unit_fn
    "12".parse::<i32>().ok().map(diverge);
    //~^ option_map_unit_fn
    Some(plus_one(1)).map(do_nothing);
    //~^ option_map_unit_fn

    // Should suggest `if let Some(_y) ...` to not override the existing foo variable
    let y = Some(42);
    y.map(do_nothing);
    //~^ option_map_unit_fn
}

fn main() {}
