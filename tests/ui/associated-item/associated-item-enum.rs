enum Enum { Variant }

impl Enum {
    const MISSPELLABLE: i32 = 0;
    fn misspellable() {}
}

trait Trait {
    fn misspellable_trait() {}
}

impl Trait for Enum {
    fn misspellable_trait() {}
}

fn main() {
    Enum::mispellable(); //~ ERROR no variant, associated function, or constant
    Enum::mispellable_trait(); //~ ERROR no variant, associated function, or constant
    Enum::MISPELLABLE; //~ ERROR no variant, associated function, or constant
}
