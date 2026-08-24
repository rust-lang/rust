//@ check-pass
//! It is very weird and mostly a compiler implementation quirk that direct_const_arg!(_) is allowed
//! to infer to a type rather than forcing it to be a constant. This test simply tracks/asserts the
//! current behavior.
#![feature(min_generic_const_args)]

struct S<T>(T);

fn main() {
    let _: S<core::direct_const_arg!(_)> = S(2u32);
    let _: S<{ core::direct_const_arg!(_) }> = S(2u32);
}
