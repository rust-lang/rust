// run-pass
fn main() {
    match (0, 0) {
        (std::usize::MIN, std::usize::MAX) => {}
        _ => {}
    }
}
