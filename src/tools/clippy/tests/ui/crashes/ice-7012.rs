//@ check-pass

#![expect(clippy::single_match)]

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
