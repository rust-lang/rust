#![allow(warnings)]

// Here the type of `c` is `Option<?T>`, where `?T` is unconstrained.
// Because there is data-flow from the `{ return; }` block, which
// diverges and hence has type `!`, into `c`, we will default `?T` to
// `!`, and hence this code compiles rather than failing and requiring
// a type annotation.

fn main() {
    let c = Some({ return; });
    c.unwrap();
}
