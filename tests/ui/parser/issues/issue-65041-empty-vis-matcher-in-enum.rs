//@ check-pass

// Here we check that a `:vis` macro matcher subsititued for the empty visibility
// (`VisibilityKind::Inherited`) is accepted when used before an enum variant.

fn main() {}

macro_rules! mac_variant {
    ($vis:vis MARKER) => {
        enum Enum {
            $vis Unit,

            $vis Tuple(u8, u16),

            $vis Struct { f: u8 },
        }
    }
}

mac_variant!(MARKER);

// We also accept visibilities on variants syntactically but not semantically.
#[cfg(false)]
enum E {
    pub U,
    pub(crate) T(u8),
    pub(super) T { f: String }
}
