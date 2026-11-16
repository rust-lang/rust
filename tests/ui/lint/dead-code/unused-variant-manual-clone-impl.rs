#![deny(dead_code)]

// Previously, the variants which are matched with `Self::...` below
// were not marked as dead.
enum Enum {
    Variant1, //~ ERROR: variants `Variant1`, `Variant3`, and `Variant5` are never constructed
    Variant2,
    Variant3(i32),
    Variant4(i32),
    Variant5 { x: i32 },
    Variant6 { x: i32 },
}

impl Clone for Enum {
    fn clone(&self) -> Enum {
        match &self {
            Enum::Variant1 => Enum::Variant1,
            Self::Variant2 => Self::Variant2,
            Enum::Variant3(b) => Enum::Variant3(*b),
            Self::Variant4(b) => Self::Variant4(*b),
            Enum::Variant5 { x } => Enum::Variant5 { x: *x },
            Self::Variant6 { x } => Self::Variant6 { x: *x },
        }
    }
}

fn main() {
    let _: Enum = todo!();
}
