#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Marker {}

trait Trait<T: Marker> {
    fn foo<U>(&self, x: U, y: T) -> (T, U) {(y, x)}
    fn bar<U: Marker>(&self, x: U) {}
}

impl<T: Marker> Trait<T> for u8 {}
impl Marker for u8 {}

fn main() {
    {
        reuse <u16 as Trait<_>>::foo;
        foo(&2, "str", 1);
        //~^ ERROR the trait bound `u16: Trait<_>` is not satisfied
    }
    {
        reuse <u16 as Trait>::foo;
        //~^ ERROR missing generics for trait `Trait`
        //~| ERROR missing generics for trait `Trait`
    }
    {
        reuse Trait::<_>::bar::<u16>;
        //~^ ERROR the trait bound `u16: Marker` is not satisfied
    }
    {
        reuse <u8 as Trait<u16>>::bar;
        //~^ ERROR the trait bound `u16: Marker` is not satisfied
    }
}
