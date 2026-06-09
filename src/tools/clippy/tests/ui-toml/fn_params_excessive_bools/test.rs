#![warn(clippy::fn_params_excessive_bools)]

fn f(_: bool) {}
fn g(_: bool, _: bool) {}
//~^ fn_params_excessive_bools

fn main() {}
