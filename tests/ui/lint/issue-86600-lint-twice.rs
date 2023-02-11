// Regression test for #86600, where an instance of the
// `illegal_floating_point_literal_pattern` lint was issued twice.

// check-pass

fn main() {
    let x = 42.0;

    match x {
        5.0 => {}
        //~^ WARNING: floating-point types cannot be used in patterns
        //~| WARNING: this was previously accepted by the compiler
        _ => {}
    }
}
