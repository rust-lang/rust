pub fn main() {
    let x: Box<_> = Box::new(1);

    let v = (1, 2);

    match v {
        (1, _) |
        (_, 2) if take(x) => (), //~ ERROR use of moved value: `x`
        _ => (),
    }
}

fn take<T>(_: T) -> bool { false }
