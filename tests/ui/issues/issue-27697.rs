//@ check-pass

use std::ops::Deref;

trait MyTrait {
    fn do_something(&self);
    fn as_str(&self) -> &str;
}

impl Deref for dyn MyTrait {
    type Target = str;
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

fn trait_object_does_something(t: &dyn MyTrait) {
    t.do_something()
}

fn main() {}
