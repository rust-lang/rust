//@ revisions: e2021 e2024

//@[e2021] edition:2021
//@[e2024] edition:2024

//@[e2021] check-pass

fn raw_gen_lt<'r#gen>() {}

fn gen_lt<'gen>() {}
//[e2024]~^ ERROR lifetimes cannot use keyword names

fn main() {}
