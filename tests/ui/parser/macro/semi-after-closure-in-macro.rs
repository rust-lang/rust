//@ check-pass

// Checks that the fix in #103222 doesn't also disqualify semicolons after
// closures within parentheses *in macros*, where they're totally allowed.

macro_rules! m {
    (($expr:expr ; )) => {
        $expr
    };
}

fn main() {
    let x = m!(( ||() ; ));
}
