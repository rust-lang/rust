//! Check that pattern types can only have specific base types

#![feature(pattern_types, const_trait_impl, pattern_type_range_trait)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

// Undoing an inner pattern type's restrictions should either be forbidden,
// or still validate correctly.
const BAD_NESTING: pattern_type!(pattern_type!(u32 is 1..) is 0..) = todo!();
//~^ ERROR: not a valid base type for range patterns
//~| ERROR: mismatched types

// We want to get the most narrowest version that a pattern could be
const BAD_NESTING2: pattern_type!(pattern_type!(i32 is 1..) is ..=-1) = todo!();
//~^ ERROR: not a valid base type for range patterns
//~| ERROR: cannot apply unary operator `-` to type `(i32) is 1..`

const BAD_NESTING3: pattern_type!(pattern_type!(i32 is 1..) is ..0) = todo!();
//~^ ERROR: not a valid base type for range patterns
//~| ERROR: not a valid base type for range patterns
//~| ERROR: mismatched types

const BAD_NESTING4: pattern_type!(() is ..0) = todo!();
//~^ ERROR: not a valid base type for range patterns
//~| ERROR: not a valid base type for range patterns
//~| ERROR: mismatched types

const BAD_NESTING5: pattern_type!(f32 is 1.0 .. 2.0) = todo!();
//~^ ERROR: not a valid base type for range patterns

fn main() {}
