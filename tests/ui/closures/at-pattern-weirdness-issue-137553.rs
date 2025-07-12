//@ edition:2024
//@ check-pass

// Background:
fn f1() {
    let mut a = (21, 37);
    // only captures a.0, example compiles fine
    let mut f = || {
        let (ref mut x, _) = a;
        *x = 42;
    };
    a.1 = 69;
    f();
}

// This used to error out:
fn f2() {
    let mut a = (21, 37);
    // used to capture all of a, now captures only a.0
    let mut f = || {
        match a {
            (ref mut x, _) => *x = 42,
        }
    };
    a.1 = 69;
    f();
}

// This was inconsistent with the following:
fn main() {
    let mut a = (21, 37);
    // the useless @-pattern would cause it to capture only a.0. now the
    // behavior is consistent with the case that doesn't use the @-pattern
    let mut f = || {
        match a {
            (ref mut x @ _, _) => *x = 42,
        }
    };
    a.1 = 69;
    f();
}
