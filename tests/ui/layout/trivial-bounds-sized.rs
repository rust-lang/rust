//@ check-pass

//! With trivial bounds, it is possible to have ADTs with unsized fields
//! in arbitrary places. Test that we do not ICE for such types.

#![feature(trivial_bounds)]
#![expect(trivial_bounds)]

struct Struct
where
    [u8]: Sized,
    [i16]: Sized,
{
    a: [u8],
    b: [i16],
    c: f32,
}

union Union
where
    [u8]: Copy,
    [i16]: Copy,
{
    a: [u8],
    b: [i16],
    c: f32,
}

enum Enum
where
    [u8]: Sized,
    [i16]: Sized,
{
    V1([u8], [i16]),
    V2([i16], f32),
}

// This forces layout computation via the `variant_size_differences` lint.
// FIXME: This could be made more robust, possibly with a variant of `rustc_layout`
// that doesn't error.
enum Check
where
    [u8]: Copy,
    [i16]: Copy,
{
    Struct(Struct),
    Union(Union),
    Enum(Enum),
}

fn main() {}
