#![feature(rustc_attrs)]
#![allow(internal_features)]

struct Cov<#[rustc_force_variance(covariant)] T>(T);
//~^ ERROR unsafe attribute used without unsafe

fn main() {}
