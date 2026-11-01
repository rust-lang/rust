// Regression test for issue #24986
// Make sure that the span of a closure marked `move` begins at the `move` keyword.

fn main() {
    let x: () = move || (); //~ ERROR mismatched types
}
