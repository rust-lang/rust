// run-rustfix
fn main() {
    // everything after `.as_ref` should be suggested
    match Some(vec![3]).as_ref().map(|v| v.as_slice()) {
        Some(vec![_x]) => (), //~ ERROR unexpected `(` after qualified path
        _ => (),
    }
}
