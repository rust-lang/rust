//@ check-pass

fn main() {
    match (0, 1, 2) {
        (pat, ..,) => {}
    }
}
