//#![feature(or_patterns)]

macro_rules! bar {
    ($nonor:pat |) => {};
}

macro_rules! baz {
    ($nonor:pat) => {};
}

fn main() {
    // Leading vert not allowed in pat<no_top_alt>
    bar!(|1| 2 | 3 |); //~ERROR: no rules expected

    // Top-level or-patterns not allowed in pat<no_top_alt>
    bar!(1 | 2 | 3 |); //~ERROR: no rules expected
    bar!(1 | 2 | 3); //~ERROR: no rules expected
    bar!((1 | 2 | 3)); //~ERROR: unexpected end
    bar!((1 | 2 | 3) |); // ok

    baz!(1 | 2 | 3); // ok
    baz!(|1| 2 | 3); // ok
    baz!(|1| 2 | 3 |); //~ERROR: expected pattern
}
