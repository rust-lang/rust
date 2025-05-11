//@ known-bug: #137190
trait Supertrait<T> {
    fn method(&self) {}
}

trait Trait<P>: Supertrait<()> {}

impl<P> Trait<P> for () {}

const fn upcast<P>(x: &dyn Trait<P>) -> &dyn Supertrait<()> {
    x
}

const fn foo() -> &'static dyn Supertrait<()> {
    upcast::<()>(&())
}

const _: &'static dyn Supertrait<()> = foo();
