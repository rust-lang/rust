// Test that we can do shifts by any integral type.

struct Panolpy {
    char: char,
    str: &'static str,
}

fn foo(p: &Panolpy) {
    22 >> p.char;
    //~^ ERROR E0277

    22 >> p.str;
    //~^ ERROR E0277

    22 >> p;
    //~^ ERROR E0277

    let x;
    22 >> x; // ambiguity error winds up being suppressed

    22 >> 1;
    // Integer literal types are OK

    // Type of the result follows the LHS, not the RHS:
    let _: i32 = 22_i64 >> 1_i32;
    //~^ ERROR mismatched types
    //~| expected `i32`, found `i64`
}

fn main() {
}
