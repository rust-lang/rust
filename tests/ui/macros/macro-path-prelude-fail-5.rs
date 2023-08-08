#[derive(Clone, Debug)] // OK
struct S;

#[derive(Debug, inline)] //~ ERROR expected derive macro, found built-in attribute `inline`
struct T;

#[derive(inline, Debug)] //~ ERROR expected derive macro, found built-in attribute `inline`
struct U;

fn main() {}
