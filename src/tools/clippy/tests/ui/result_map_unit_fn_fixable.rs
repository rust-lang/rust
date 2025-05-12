#![warn(clippy::result_map_unit_fn)]
#![allow(unused)]
#![allow(clippy::uninlined_format_args)]

fn do_nothing<T>(_: T) {}

fn diverge<T>(_: T) -> ! {
    panic!()
}

fn plus_one(value: usize) -> usize {
    value + 1
}

struct HasResult {
    field: Result<usize, usize>,
}

impl HasResult {
    fn do_result_nothing(&self, value: usize) {}

    fn do_result_plus_one(&self, value: usize) -> usize {
        value + 1
    }
}

#[rustfmt::skip]
fn result_map_unit_fn() {
    let x = HasResult { field: Ok(10) };

    x.field.map(plus_one);
    let _: Result<(), usize> = x.field.map(do_nothing);

    x.field.map(do_nothing);
    //~^ result_map_unit_fn

    x.field.map(do_nothing);
    //~^ result_map_unit_fn

    x.field.map(diverge);
    //~^ result_map_unit_fn

    let captured = 10;
    if let Ok(value) = x.field { do_nothing(value + captured) };
    let _: Result<(), usize> = x.field.map(|value| do_nothing(value + captured));

    x.field.map(|value| x.do_result_nothing(value + captured));
    //~^ result_map_unit_fn

    x.field.map(|value| { x.do_result_plus_one(value + captured); });
    //~^ result_map_unit_fn


    x.field.map(|value| do_nothing(value + captured));
    //~^ result_map_unit_fn

    x.field.map(|value| { do_nothing(value + captured) });
    //~^ result_map_unit_fn

    x.field.map(|value| { do_nothing(value + captured); });
    //~^ result_map_unit_fn

    x.field.map(|value| { { do_nothing(value + captured); } });
    //~^ result_map_unit_fn


    x.field.map(|value| diverge(value + captured));
    //~^ result_map_unit_fn

    x.field.map(|value| { diverge(value + captured) });
    //~^ result_map_unit_fn

    x.field.map(|value| { diverge(value + captured); });
    //~^ result_map_unit_fn

    x.field.map(|value| { { diverge(value + captured); } });
    //~^ result_map_unit_fn


    x.field.map(|value| plus_one(value + captured));
    x.field.map(|value| { plus_one(value + captured) });
    x.field.map(|value| { let y = plus_one(value + captured); });
    //~^ result_map_unit_fn

    x.field.map(|value| { plus_one(value + captured); });
    //~^ result_map_unit_fn

    x.field.map(|value| { { plus_one(value + captured); } });
    //~^ result_map_unit_fn


    x.field.map(|ref value| { do_nothing(value + captured) });
    //~^ result_map_unit_fn

    x.field.map(|value| println!("{:?}", value));
    //~^ result_map_unit_fn
}

fn main() {}
