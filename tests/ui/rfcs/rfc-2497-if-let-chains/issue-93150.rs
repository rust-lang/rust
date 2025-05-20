//@ run-pass
fn main() {
    match true {
        _ if let true = true && true => {}
        _ => {}
    }
}
