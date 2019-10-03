use std::ops::Deref;
trait Trait {}

struct Struct;

impl Deref for Struct {
    type Target = dyn Trait;
    fn deref(&self) -> &dyn Trait {
        unimplemented!();
    }
}
//~^^^^ ERROR `impl` item doesn't match `trait` item

fn main() {}
