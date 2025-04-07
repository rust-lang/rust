//@ check-pass

#![allow(clippy::unnecessary_literal_unwrap)]

trait IsErr {
    fn is_err(&self, err: &str) -> bool;
}

impl<T> IsErr for Option<T> {
    fn is_err(&self, _err: &str) -> bool {
        true
    }
}

fn main() {
    let t = Some(1);

    if t.is_err("") {
        t.unwrap();
    }
}
