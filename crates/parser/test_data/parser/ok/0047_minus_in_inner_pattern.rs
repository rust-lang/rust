// https://github.com/rust-analyzer/rust-analyzer/issues/972

fn main() {
    match Some(-1) {
        Some(-1) => (),
        _ => (),
    }

    match Some((-1, -1)) {
        Some((-1, -1)) => (),
        _ => (),
    }

    match A::B(-1, -1) {
        A::B(-1, -1) => (),
        _ => (),
    }

    if let Some(-1) = Some(-1) {
    }
}

enum A {
    B(i8, i8)
}

fn foo(-128..=127: i8) {}
