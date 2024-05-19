// Ensure that we don't get a cycle error from trying to determine whether an
// opaque type implements `Freeze` in safety checking, when it doesn't matter.

//@ check-pass

#![feature(rustc_attrs)]

struct AnyValue<T>(T);

// No need to check for `Freeze` here, there's no
// `rustc_layout_scalar_valid_range_start` involved.
fn not_restricted(c: bool) -> impl Sized {
    if c {
        let x = AnyValue(not_restricted(false));
        &x.0;
    }
    2u32
}

#[rustc_layout_scalar_valid_range_start(1)]
struct NonZero<T>(T);

// No need to check for `Freeze` here, we're not borrowing the field.
fn not_field(c: bool) -> impl Sized {
    if c {
        let x = unsafe { NonZero(not_field(false)) };
        &x;
    }
    5u32
}

fn main() {}
