// run-pass

//#![feature(or_patterns)]

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

macro_rules! quux {
    ($orpat1:pat | $orpat2:pat, $val:expr) => {
        match $val {
            $orpat1 => $val,
            _ => B(0xDEADBEEFu64),
        }
    };
}

macro_rules! baz {
    ($orpat:pat, $val:expr) => {
        match $val {
            $orpat => 42,
            _ => 0xDEADBEEFu64,
        }
    };
    ($nonor:pat | $val:expr, C) => {
        match $val {
            x @ ($orpat) => x,
            _ => 0xDEADBEEFu64,
        }
    };
}

macro_rules! foo2a {
    ($orpat:pat, $val:expr) => {
        match $val {
            x @ ($orpat) => x,
            _ => 0xDEADBEEFu64,
        }
    };
}

macro_rules! foo2b {
    ($nonor:pat | $val:expr, 2) => {
        match $val {
            x @ $nonor => x,
            _ => 0xDEADBEEFu64,
        }
    };
}

macro_rules! foo2 {
    ($orpat:pat, $val:expr) => {
        match $val {
            x @ ($orpat) => x,
            _ => 0xDEADBEEFu64,
        }
    };
    ($nonor:pat | $val:expr, 2) => {
        match $val {
            x @ $nonor => x,
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

    // Leading vert in or-pattern makes baz! unambiguous.
    let y = baz!(|C| D, C);
    assert_eq!(y, 42u64);

    // Or-separator fallback.
    let y = quux!(C | D, D);
    assert_eq!(y, B(0xDEADBEEFu64));

    // Test tricky ambiguous case.
    let y = foo2a!(1 | 2, 2);
    assert_eq!(y, 2);
    let y = foo2b!(1 | 2, 2);
    assert_eq!(y, 0xDEADBEEFu64);
    let y = foo2!(1 | 2, 2);
    assert_eq!(y, 2);
}
