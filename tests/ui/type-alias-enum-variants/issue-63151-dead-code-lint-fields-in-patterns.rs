//@ check-pass

// Regression test for the issue #63151:
// Spurious unused field warning when matching variants under a `Self` scope
//
// This test checks that the `dead_code` lint properly inspects fields
// in struct patterns that use a type relative path.

#![deny(dead_code)]

enum Enum {
    Variant { field: usize }
}

impl Enum {
    fn read_field(self) -> usize {
        match self {
            Self::Variant { field } => field
        }
    }
}

fn main() {
    let e = Enum::Variant { field: 42 };
    println!("{}", e.read_field());
}
