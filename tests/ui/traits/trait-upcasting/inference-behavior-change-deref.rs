#![deny(deref_into_dyn_supertrait)]
#![feature(trait_upcasting)] // remove this and the test compiles

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
    //~^ ERROR mismatched types
}
