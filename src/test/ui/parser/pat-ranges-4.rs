// Parsing of range patterns

// compile-flags: -Z parse-only

fn main() {
    let 10 - 3 ..= 10 = 8;
    //~^ error: expected one of `...`, `..=`, `..`, `:`, `;`, or `=`, found `-`
}
