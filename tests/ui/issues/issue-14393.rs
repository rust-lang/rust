//@ run-pass

fn main() {
    match ("", 1_usize) {
        (_, 42_usize) => (),
        ("", _) => (),
        _ => ()
    }
}
