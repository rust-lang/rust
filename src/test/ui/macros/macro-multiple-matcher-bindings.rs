// Test that duplicate matcher binding names are caught at declaration time, rather than at macro
// invocation time.
//
// FIXME(mark-i-m): Update this when it becomes a hard error.

// compile-pass

#![allow(unused_macros)]
#![warn(duplicate_matcher_binding_name)]

macro_rules! foo1 {
    ($a:ident, $a:ident) => {}; //~WARNING duplicate matcher binding
    ($a:ident, $a:path) => {};  //~WARNING duplicate matcher binding
}

macro_rules! foo2 {
    ($a:ident) => {}; // OK
    ($a:path) => {};  // OK
}

macro_rules! foo3 {
    ($a:ident, $($a:ident),*) => {}; //~WARNING duplicate matcher binding
    ($($a:ident)+ # $($($a:path),+);*) => {}; //~WARNING duplicate matcher binding
}

fn main() {}
