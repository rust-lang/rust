// When a MULTI-character string literal is used where a char should be,
// DO NOT suggest changing to single quotes.

fn main() {
    let _: char = "foo"; //~ ERROR mismatched types
}
