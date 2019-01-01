// run-pass

fn main() {
    match b"    " {
        b"1234" => {},
        _ => {},
    }
}
