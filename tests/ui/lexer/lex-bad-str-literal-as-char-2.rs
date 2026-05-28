//@ run-rustfix
fn main() {
    println!(' 1 + 1'); //~ ERROR character literal may only contain one codepoint
}
