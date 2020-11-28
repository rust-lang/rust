// run-pass

//#![feature(or_patterns)]

macro_rules! foo {
    ($orpat:pat, $val:expr) => {
        match $val {
            x @ ($orpat) => x,
            _ => 0xDEADBEEF,
        }
    };
    ($nonor:pat | $val:expr, 3) => {
        match $val {
            x @ ($orpat) => x,
            _ => 0xDEADBEEF,
        }
    };
}

macro_rules! bar {
    ($nonor:pat |) => {};
}

macro_rules! baz {
    ($nonor:pat) => {};
}

fn main() {
    // Test ambiguity.
    foo!(1 | 2, 3); //~ERROR: multiple matchers

    // Leading vert not allowed in pat<no_top_alt>
    bar!(1 | 2 | 3 |); // ok
    bar!(|1| 2 | 3 |); //~ERROR: no rules expected
    bar!(1 | 2 | 3); //~ERROR: unexpected end

    baz!(1 | 2 | 3); // ok
    baz!(|1| 2 | 3); // ok
    baz!(|1| 2 | 3 |); //~ERROR: no rules expected
}
