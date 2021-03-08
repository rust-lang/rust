// run-pass
// edition:2021
// ignore-test
// FIXME(mark-i-m): enable this test again when 2021 machinery is available

#![feature(or_patterns)]

use Foo::*;

#[derive(Eq, PartialEq, Debug)]
enum Foo {
    A(u64),
    B(u64),
    C,
    D,
}

macro_rules! foo {
    ($orpat:pat, $val:expr) => {
        match $val {
            x @ ($orpat) => x, // leading vert would not be allowed in $orpat
            _ => B(0xDEADBEEFu64),
        }
    };
}

macro_rules! bar {
    ($orpat:pat, $val:expr) => {
        match $val {
            $orpat => 42, // leading vert allowed here
            _ => 0xDEADBEEFu64,
        }
    };
}

fn main() {
    // Test or-pattern.
    let y = foo!(A(_)|B(_), A(32));
    assert_eq!(y, A(32));

    // Leading vert in or-pattern.
    let y = bar!(|C| D, C);
    assert_eq!(y, 42u64);
}
