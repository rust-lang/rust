//@ run-pass
fn main() {
    match (0, 0) {
        (usize::MIN, usize::MAX) => {}
        _ => {}
    }
}
