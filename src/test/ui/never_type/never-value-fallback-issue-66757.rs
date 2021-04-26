// Regression test for #66757
//
// Test than when you have a `!` value (e.g., the local variable
// never) and an uninferred variable (here the argument to `From`) it
// doesn't fallback to `()` but rather `!`.
//
// run-pass

struct E;

impl From<!> for E {
    fn from(_: !) -> E {
        E
    }
}

#[allow(unreachable_code)]
#[allow(dead_code)]
fn foo(never: !) {
    <E as From<!>>::from(never);  // Ok
    <E as From<_>>::from(never);  // Inference fails here
}

fn main() { }
