// Parsing of range patterns

// compile-flags: -Z parse-only

fn main() {
    let macropus!() ..= 11 = 12; //~ error: expected one of `:`, `;`, or `=`, found `..=`
}
