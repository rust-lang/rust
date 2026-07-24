//! Test that using `#[arg_splat]` on maybe-tuple generic function arguments is an error,
//! but only when the generics aren't tuples.

#![allow(incomplete_features)]
#![feature(arg_splat)]
#![expect(unused)]

fn unbound_generic_arg<T>(#[arg_splat] t: T) {} //~ ERROR cannot use splat attribute; the splatted argument type must be a tuple or unit, not a u32

fn main() {
    unbound_generic_arg();
    unbound_generic_arg::<()>();

    unbound_generic_arg(1);
    unbound_generic_arg::<(u32,)>(1);

    unbound_generic_arg(1, 2.0);
    unbound_generic_arg::<(u32, f32)>(1, 2.0);

    // The error comes from this call
    unbound_generic_arg::<u32>(1);
}
