#![deny(elided_lifetimes_in_paths)]

// Previously, the elided-lifetimes-in-path lint would fire, but we don't want
// that, because `'_` isn't legal in struct declarations.

struct Betrayal<'a> { x: &'a u8 }

struct Heartbreak(Betrayal);  //~ ERROR missing lifetime specifier

fn main() {}
