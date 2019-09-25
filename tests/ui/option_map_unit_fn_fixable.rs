#![warn(clippy::option_map_unit_fn)]
#![allow(unused)]

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

impl HasOption {
    fn do_option_nothing(self: &Self, value: usize) {}

    fn do_option_plus_one(self: &Self, value: usize) -> usize {
        value + 1
    }
}
#[rustfmt::skip]
fn option_map_unit_fn() {
    let x = HasOption { field: Some(10) };

    x.field.map(plus_one);
    let _ : Option<()> = x.field.map(do_nothing);

    x.field.map(do_nothing);

    x.field.map(do_nothing);

    x.field.map(diverge);

    let captured = 10;
    if let Some(value) = x.field { do_nothing(value + captured) };
    let _ : Option<()> = x.field.map(|value| do_nothing(value + captured));

    x.field.map(|value| x.do_option_nothing(value + captured));

    x.field.map(|value| { x.do_option_plus_one(value + captured); });


    x.field.map(|value| do_nothing(value + captured));

    x.field.map(|value| { do_nothing(value + captured) });

    x.field.map(|value| { do_nothing(value + captured); });

    x.field.map(|value| { { do_nothing(value + captured); } });


    x.field.map(|value| diverge(value + captured));

    x.field.map(|value| { diverge(value + captured) });

    x.field.map(|value| { diverge(value + captured); });

    x.field.map(|value| { { diverge(value + captured); } });


    x.field.map(|value| plus_one(value + captured));
    x.field.map(|value| { plus_one(value + captured) });
    x.field.map(|value| { let y = plus_one(value + captured); });

    x.field.map(|value| { plus_one(value + captured); });

    x.field.map(|value| { { plus_one(value + captured); } });


    x.field.map(|ref value| { do_nothing(value + captured) });


    x.field.map(|value| { do_nothing(value); do_nothing(value) });

    x.field.map(|value| if value > 0 { do_nothing(value); do_nothing(value) });

    // Suggestion for the let block should be `{ ... }` as it's too difficult to build a
    // proper suggestion for these cases
    x.field.map(|value| {
        do_nothing(value);
        do_nothing(value)
    });
    x.field.map(|value| { do_nothing(value); do_nothing(value); });

    // The following should suggest `if let Some(_X) ...` as it's difficult to generate a proper let variable name for them
    Some(42).map(diverge);
    "12".parse::<i32>().ok().map(diverge);
    Some(plus_one(1)).map(do_nothing);

    // Should suggest `if let Some(_y) ...` to not override the existing foo variable
    let y = Some(42);
    y.map(do_nothing);
}

fn main() {}
