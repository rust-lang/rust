//@ run-pass

fn promote<const N: i32>() {
    let _ = &N;
}

fn main() {
    promote::<0>();
}
