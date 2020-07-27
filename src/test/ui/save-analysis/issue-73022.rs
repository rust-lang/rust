// build-pass
// compile-flags: -Zsave-analysis
enum Enum2 {
    Variant8 { _field: bool },
}

impl Enum2 {
    fn new_variant8() -> Enum2 {
        Self::Variant8 { _field: true }
    }
}

fn main() {}
