//@ check-pass
//@ edition: 2024
//
// Check that we don't warn on `as` casts of never to any as unreachable.
// While they *are* unreachable, sometimes they are required to appeal typeck.
#![deny(unreachable_code)]

fn a() {
    _ = {return} as u32;
}

fn b() {
    (return) as u32;
}

// example that needs an explicit never-to-any `as` cast
fn example() -> impl Iterator<Item = u8> {
    todo!() as std::iter::Empty<_>
}

fn main() {}
