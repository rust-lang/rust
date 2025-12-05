//! Test type errors for byte string literal patterns. `deref_patterns` allows byte string literal
//! patterns to have type `[u8]` or `[u8; N]` when matching on a slice or array; this can affect the
//! "found" type reported in error messages when matching on a slice or array of the wrong type.

//@ dont-require-annotations: NOTE

#![feature(deref_patterns)]
#![expect(incomplete_features)]

fn main() {
    // Baseline 1: under normal circumstances, byte string literal patterns have type `&[u8; N]`,
    // the same as byte string literals.
    if let b"test" = () {}
    //~^ ERROR mismatched types
    //~| NOTE expected `()`, found `&[u8; 4]`

    // Baseline 2: there's a special case for byte string patterns in stable rust, allowing them to
    // match on slice references. This affects the error when matching on a non-`&[u8]` slice ref,
    // reporting the "found" type as `&[u8]`.
    if let b"test" = &[] as &[i8] {}
    //~^ ERROR mismatched types
    //~| NOTE expected `&[i8]`, found `&[u8]`

    // Test matching on a non-`[u8]` slice: the pattern has type `[u8]` if a slice is expected.
    if let b"test" = *(&[] as &[i8]) {}
    //~^ ERROR mismatched types
    //~| NOTE expected `[i8]`, found `[u8]`

    // Test matching on a non-`[u8;4]` array: the pattern has type `[u8;4]` if an array is expected.
    if let b"test" = [()] {}
    //~^ ERROR mismatched types
    //~| NOTE expected `[(); 1]`, found `[u8; 4]`
    if let b"test" = *b"this array is too long" {}
    //~^ ERROR mismatched types
    //~| NOTE expected an array with a size of 22, found one with a size of 4

    // Test matching on `&mut T`: we peel the `&mut` before applying the usual special cases.
    // No special cases apply to `()`, so the "found" type is the type of the literal.
    if let b"test" = &mut () {}
    //~^ ERROR mismatched types
    //~| NOTE expected `()`, found `&[u8; 4]`

    // If the pointee is an array or slice, the usual special cases will apply to the "found" type:
    if let b"test" = &mut [] as &mut [i8] {}
    //~^ ERROR mismatched type
    //~| NOTE expected `[i8]`, found `[u8]`

    if let b"test" = &mut [()] {}
    //~^ ERROR mismatched types
    //~| NOTE expected `[(); 1]`, found `[u8; 4]`

    if let b"test" = &mut *b"this array is too long" {}
    //~^ ERROR mismatched type
    //~| NOTE expected an array with a size of 22, found one with a size of 4
}
