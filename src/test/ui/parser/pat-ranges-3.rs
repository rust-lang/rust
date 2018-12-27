// Parsing of range patterns

// compile-flags: -Z parse-only

fn main() {
    let 10 ..= 10 + 3 = 12; //~ expected one of `:`, `;`, or `=`, found `+`
}
