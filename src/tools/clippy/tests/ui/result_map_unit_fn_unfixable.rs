#![warn(clippy::result_map_unit_fn)]
#![feature(never_type)]
#![allow(unused, clippy::unnecessary_map_on_constructor)]
//@no-rustfix
struct HasResult {
    field: Result<usize, usize>,
}

fn do_nothing<T>(_: T) {}

fn diverge<T>(_: T) -> ! {
    panic!()
}

fn plus_one(value: usize) -> usize {
    value + 1
}

#[rustfmt::skip]
fn result_map_unit_fn() {
    let x = HasResult { field: Ok(10) };

    x.field.map(|value| { do_nothing(value); do_nothing(value) });
    //~^ ERROR: called `map(f)` on an `Result` value where `f` is a closure that returns t
    //~| NOTE: `-D clippy::result-map-unit-fn` implied by `-D warnings`

    x.field.map(|value| if value > 0 { do_nothing(value); do_nothing(value) });
    //~^ ERROR: called `map(f)` on an `Result` value where `f` is a closure that returns t

    // Suggestion for the let block should be `{ ... }` as it's too difficult to build a
    // proper suggestion for these cases
    x.field.map(|value| {
    //~^ ERROR: called `map(f)` on an `Result` value where `f` is a closure that returns t
        do_nothing(value);
        do_nothing(value)
    });
    x.field.map(|value| { do_nothing(value); do_nothing(value); });
    //~^ ERROR: called `map(f)` on an `Result` value where `f` is a closure that returns t

    // The following should suggest `if let Ok(_X) ...` as it's difficult to generate a proper let variable name for them
    let res: Result<!, usize> = Ok(42).map(diverge);
    "12".parse::<i32>().map(diverge);
    //~^ ERROR: called `map(f)` on an `Result` value where `f` is a function that returns

    let res: Result<(), usize> = Ok(plus_one(1)).map(do_nothing);

    // Should suggest `if let Ok(_y) ...` to not override the existing foo variable
    let y: Result<usize, usize> = Ok(42);
    y.map(do_nothing);
    //~^ ERROR: called `map(f)` on an `Result` value where `f` is a function that returns
}

fn main() {}
