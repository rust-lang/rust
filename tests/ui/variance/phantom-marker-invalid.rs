#![feature(rustc_attrs)]
#![allow(internal_features)]

#[rustc_phantom_marker]
//~^ ERROR `rustc_phantom_marker` can only be used on a unit struct
struct Tuple(u8);

#[rustc_phantom_marker]
//~^ ERROR `rustc_phantom_marker` can only be used on a unit struct
struct EmptyTuple();

#[rustc_phantom_marker]
//~^ ERROR `rustc_phantom_marker` can only be used on a unit struct
struct Named { field: u8 }

#[rustc_phantom_marker]
//~^ ERROR `rustc_phantom_marker` can only be used on a unit struct
struct EmptyNamed {}

#[rustc_phantom_marker]
//~^ ERROR the `rustc_phantom_marker` attribute cannot be used on enums
enum Enum { Unit }

#[rustc_phantom_marker]
//~^ ERROR the `rustc_phantom_marker` attribute cannot be used on unions
union Union { field: u8 }

#[rustc_phantom_marker]
//~^ ERROR the `rustc_phantom_marker` attribute cannot be used on type aliases
type Alias = ();

#[rustc_phantom_marker]
//~^ ERROR the `rustc_phantom_marker` attribute cannot be used on functions
fn function() {}

fn main() {}
