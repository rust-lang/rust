#![deny(deref_into_dyn_supertrait)]
use std::ops::Deref;

trait Bar<T> {}
impl<T, U> Bar<U> for T {}

trait Foo: Bar<i32> {
    fn as_dyn_bar_u32<'a>(&self) -> &(dyn Bar<u32> + 'a);
}

impl Foo for () {
    fn as_dyn_bar_u32<'a>(&self) -> &(dyn Bar<u32> + 'a) {
        self
    }
}

impl<'a> Deref for dyn Foo + 'a {
    //~^ ERROR this `Deref` implementation is covered by an implicit supertrait coercion
    //~| WARN this will change its meaning in a future release!
    type Target = dyn Bar<u32> + 'a;

    fn deref(&self) -> &Self::Target {
        self.as_dyn_bar_u32()
    }
}

fn take_dyn<T>(x: &dyn Bar<T>) -> T {
    todo!()
}

fn main() {
    let x: &dyn Foo = &();
    let y = take_dyn(x);
    let z: u32 = y;
}
