// run-pass
// pretty-expanded FIXME #23616

fn main() {
    match ("", 1_usize) {
        (_, 42_usize) => (),
        ("", _) => (),
        _ => ()
    }
}
