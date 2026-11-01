// https://github.com/rust-lang/rust/issues/93150
//@ run-pass

fn main() {
    match true {
        _ if let true = true && true => {}
        _ => {}
    }
}
