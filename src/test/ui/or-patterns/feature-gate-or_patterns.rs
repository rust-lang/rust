fn main() {}

pub fn example(x: Option<usize>) {
    match x {
        Some(0 | 1 | 2) => {}
        //~^ ERROR: or-patterns syntax is experimental
        _ => {}
    }
}

// Test the `pat` macro fragment parser:
macro_rules! accept_pat {
    ($p:pat) => {}
}

accept_pat!((p | q)); //~ ERROR or-patterns syntax is experimental
accept_pat!((p | q,)); //~ ERROR or-patterns syntax is experimental
accept_pat!(TS(p | q)); //~ ERROR or-patterns syntax is experimental
accept_pat!(NS { f: p | q }); //~ ERROR or-patterns syntax is experimental
accept_pat!([p | q]); //~ ERROR or-patterns syntax is experimental

// Non-macro tests:

#[cfg(FALSE)]
fn or_patterns() {
    // Gated:

    let | A | B; //~ ERROR or-patterns syntax is experimental
    //~^ ERROR top-level or-patterns are not allowed
    let A | B; //~ ERROR or-patterns syntax is experimental
    //~^ ERROR top-level or-patterns are not allowed
    for | A | B in 0 {} //~ ERROR or-patterns syntax is experimental
    for A | B in 0 {} //~ ERROR or-patterns syntax is experimental
    fn fun((A | B): _) {} //~ ERROR or-patterns syntax is experimental
    let _ = |(A | B): u8| (); //~ ERROR or-patterns syntax is experimental
    let (A | B); //~ ERROR or-patterns syntax is experimental
    let (A | B,); //~ ERROR or-patterns syntax is experimental
    let A(B | C); //~ ERROR or-patterns syntax is experimental
    let E::V(B | C); //~ ERROR or-patterns syntax is experimental
    let S { f1: B | C, f2 }; //~ ERROR or-patterns syntax is experimental
    let E::V { f1: B | C, f2 }; //~ ERROR or-patterns syntax is experimental
    let [A | B]; //~ ERROR or-patterns syntax is experimental

    // Top level of `while`, `if`, and `match` arms are allowed:

    while let | A = 0 {}
    while let A | B = 0 {}
    if let | A = 0 {}
    if let A | B = 0 {}
    match 0 {
        | A => {},
        A | B => {},
    }
}
