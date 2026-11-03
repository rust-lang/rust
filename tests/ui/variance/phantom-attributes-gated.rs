struct Cov<#[unsafe(rustc_force_variance(covariant))] T>(T);
//~^ ERROR use of an internal attribute

#[rustc_phantom_marker]
//~^ ERROR use of an internal attribute
struct Marker;

fn main() {}
