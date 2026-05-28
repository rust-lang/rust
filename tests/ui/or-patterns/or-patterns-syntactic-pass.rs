// Here we test all the places `|` is *syntactically* allowed.
// This is not a semantic test. We only test parsing.

//@ check-pass

fn main() {}

// Test the `pat` macro fragment parser:
macro_rules! accept_pat {
    ($p:pat) => {};
}

accept_pat!((p | q));
accept_pat!((p | q,));
accept_pat!(TS(p | q));
accept_pat!(NS { f: p | q });
accept_pat!([p | q]);

// Non-macro tests:

#[cfg(false)]
fn or_patterns() {
    // Top level of `let`:
    let (| A | B);
    let (A | B);
    let (A | B): u8;
    let (A | B) = 0;
    let (A | B): u8 = 0;

    // Top level of `for`:
    for | A | B in 0 {}
    for A | B in 0 {}

    // Top level of `while`:
    while let | A | B = 0 {}
    while let A | B = 0 {}

    // Top level of `if`:
    if let | A | B = 0 {}
    if let A | B = 0 {}

    // Top level of `match` arms:
    match 0 {
        | A | B => {}
        A | B => {}
    }

    // Functions:
    fn fun((A | B): _) {}

    // Lambdas:
    let _ = |(A | B): u8| ();

    // Parenthesis and tuple patterns:
    let (A | B);
    let (A | B,);

    // Tuple struct patterns:
    let A(B | C);
    let E::V(B | C);

    // Struct patterns:
    let S { f1: B | C, f2 };
    let E::V { f1: B | C, f2 };

    // Slice patterns:
    let [A | B, .. | ..];

    // These bind as `(prefix p) | q` as opposed to `prefix (p | q)`:
    let (box 0 | 1); // Unstable; we *can* change the precedence if we want.
                     //~^ WARN box pattern syntax is experimental
                     //~| WARN unstable syntax
    let (&0 | 1);
    let (&mut 0 | 1);
    let (x @ 0 | 1);
    let (ref x @ 0 | 1);
    let (ref mut x @ 0 | 1);
}
