use std::ops::Deref;
trait Trait {}

struct Struct;

impl Deref for Struct {
    type Target = dyn Trait;
    fn deref(&self) -> &dyn Trait {
        unimplemented!();
    }
}
//~^^^^ ERROR cannot infer an appropriate lifetime for lifetime parameter

fn main() {}
