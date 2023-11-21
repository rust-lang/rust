// check-pass

use std::ops::Deref;

trait Bar<'a> {}
trait Foo<'a>: Bar<'a> {}

impl<'a> Deref for dyn Foo<'a> {
    //~^ WARN this `Deref` implementation is covered by an implicit supertrait coercion
    type Target = dyn Bar<'a>;

    fn deref(&self) -> &Self::Target {
        todo!()
    }
}

fn main() {}
