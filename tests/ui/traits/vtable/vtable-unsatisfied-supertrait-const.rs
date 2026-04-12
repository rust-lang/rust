// Regression test for #154073.
// Verify that we don't ICE when building vtable entries
// for a trait whose supertrait is not implemented, during
// const evaluation of a static initializer.

//@ compile-flags: --crate-type lib

trait Bar: Send + Sync + Droppable {}

impl<T: Send> Bar for T {}
//~^ ERROR the trait bound `T: Droppable` is not satisfied
//~| ERROR `T` cannot be shared between threads safely

trait Droppable {
    fn drop(&self);
}

const fn upcast(x: &dyn Bar) -> &(dyn Send + Sync) {
    x
}

static BAR: &(dyn Send + Sync) = upcast(&false);
