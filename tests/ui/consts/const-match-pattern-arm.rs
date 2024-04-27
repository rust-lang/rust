//@ check-pass

const _: bool = match Some(true) {
    Some(value) => true,
    _ => false
};

const _: bool = {
    match Some(true) {
        Some(value) => true,
        _ => false
    }
};

fn main() {}
