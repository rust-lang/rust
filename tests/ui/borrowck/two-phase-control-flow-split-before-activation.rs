//@ run-pass

fn main() {
    let mut a = 0;
    let mut b = 0;
    let p = if maybe() {
        &mut a
    } else {
        &mut b
    };
    use_(p);
}

fn maybe() -> bool { false }
fn use_<T>(_: T) { }
