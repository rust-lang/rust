#![deny(dead_code)]

enum Enum {
    Variant1,
    Variant2(i32), //~ ERROR: `Variant2` is never constructed
}

impl Clone for Enum {
    fn clone(&self) -> Self {
        match (self,) {
            (Self::Variant1,) => Self::Variant1,
            (Self::Variant2(value),) => Self::Variant2(*value),
        }
    }
}

fn main() {
    let _ = Enum::Variant1;
}
