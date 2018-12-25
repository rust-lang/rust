// Parsing of range patterns

// compile-flags: -Z parse-only

fn main() {
    let 10 ..= makropulos!() = 12; //~ error: expected one of `::`, `:`, `;`, or `=`, found `!`
}
