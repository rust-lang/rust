//@ check-pass

#![allow(clippy::all)]

enum _MyOption {
    None,
    Some(()),
}

impl _MyOption {
    fn _foo(&self) {
        match self {
            &Self::Some(_) => {},
            _ => {},
        }
    }
}

fn main() {}
