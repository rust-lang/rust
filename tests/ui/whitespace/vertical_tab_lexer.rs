// This test checks that the Rust lexer treats a vertical tab (\x0B)
// as whitespace. That matches how the language parses code, even though
// the standard library's is_ascii_whitespace does not include it.
//
// See: https://github.com/rust-lang/rust-project-goals/issues/53

fn main() {
    let x = 5;
    let y = 10;
    let z = x + y;
    println!("{}", z);
}