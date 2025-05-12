//@ known-bug: #136661

#![allow(unused)]

trait Supertrait<T> {}

trait Other {
    fn method(&self) {}
}

impl WithAssoc for &'static () {
    type As = ();
}

trait WithAssoc {
    type As;
}

trait Trait<P: WithAssoc>: Supertrait<P::As> {
    fn method(&self) {}
}

fn hrtb<T: for<'a> Trait<&'a ()>>() {}

pub fn main() {}
