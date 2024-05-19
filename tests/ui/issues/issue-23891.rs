//@ run-pass
macro_rules! id {
    ($s: pat) => ($s);
}

fn main() {
    match (Some(123), Some(456)) {
        (id!(Some(a)), _) | (_, id!(Some(a))) => println!("{}", a),
        _ => (),
    }
}
