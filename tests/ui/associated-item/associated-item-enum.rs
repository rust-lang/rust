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
    Enum::mispellable(); //~ ERROR no variant or associated item
    Enum::mispellable_trait(); //~ ERROR no variant or associated item
    Enum::MISPELLABLE; //~ ERROR no variant or associated item
}
