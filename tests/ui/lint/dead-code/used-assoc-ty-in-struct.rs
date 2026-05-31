//@ check-pass

#![deny(dead_code)]

trait Trait { type Ty; }

impl Trait for () { type Ty = (); }

pub struct Wrap(Inner<()>);
struct Inner<T: Trait>(T::Ty); // <- use of QPath::TypeRelative `Ty` in a non-body only

impl Wrap {
    pub fn live(self) { _ = self.0; }
}

fn main() {}
