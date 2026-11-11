//@ run-pass

// https://github.com/rust-lang/rust/issues/27918

fn main() {
    match b"    " {
        b"1234" => {},
        _ => {},
    }
}
