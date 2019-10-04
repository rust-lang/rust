use std::ops::Deref;
trait Trait {}

struct Struct;

impl Deref for Struct {
    type Target = dyn Trait;
    fn deref(&self) -> &dyn Trait {
        unimplemented!();
    }
}
//~^^^^ ERROR `impl` item signature doesn't match `trait` item signature

fn main() {}
