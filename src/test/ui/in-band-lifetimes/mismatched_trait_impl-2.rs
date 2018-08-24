use std::ops::Deref;
trait Trait {}

struct Struct;

impl Deref for Struct {
    type Target = Trait;
    fn deref(&self) -> &Trait {
        unimplemented!();
    }
}
//~^^^^ ERROR cannot infer an appropriate lifetime for lifetime parameter

fn main() {}
