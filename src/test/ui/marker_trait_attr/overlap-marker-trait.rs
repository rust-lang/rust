// Test for RFC 1268: we allow overlapping impls of marker traits,
// that is, traits with #[marker]. In this case, a type `T` is
// `MyMarker` if it is either `Debug` or `Display`. This test just
// checks that we don't consider **all** types to be `MyMarker`.

#![feature(marker_trait_attr)]

use std::fmt::{Debug, Display};

#[marker] trait Marker {}

impl<T: Debug> Marker for T {}
impl<T: Display> Marker for T {}

fn is_marker<T: Marker>() { }

struct NotDebugOrDisplay;

fn main() {
    // Debug && Display:
    is_marker::<i32>();

    // Debug && !Display:
    is_marker::<Vec<i32>>();

    // !Debug && !Display
    is_marker::<NotDebugOrDisplay>(); //~ ERROR
}
