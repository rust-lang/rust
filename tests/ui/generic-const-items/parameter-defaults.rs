#![feature(generic_const_items)]
#![allow(incomplete_features)]

// Check that we emit a *hard* error (not just a lint warning or error for example) for generic
// parameter defaults on free const items since we are not limited by backward compatibility.
#![allow(invalid_type_param_default)] // Should have no effect here.

// FIXME(default_type_parameter_fallback): Consider reallowing them once they work properly.

const NONE<T = ()>: Option<T> = None::<T>;
//~^ ERROR defaults for generic parameters are not allowed here

impl Host {
    const NADA<T = ()>: Option<T> = None::<T>;
    //~^ ERROR defaults for generic parameters are not allowed here
}

enum Host {}

fn body0() { let _ = NONE; } //~ ERROR type annotations needed
fn body1() { let _ = Host::NADA; } //~ ERROR type annotations needed

fn main() {}
