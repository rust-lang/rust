//@ known-bug: #137190
trait Supertrait {
    fn method(&self) {}
}

trait Trait: Supertrait {}

impl Trait for () {}

const _: &dyn Supertrait = &() as &dyn Trait as &dyn Supertrait;
