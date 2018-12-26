trait Foo {
    fn dummy(&self) { }
}

pub trait Bar : Foo {}
//~^ ERROR private trait `Foo` in public interface [E0445]
//~| NOTE can't leak private trait
pub struct Bar2<T: Foo>(pub T);
//~^ ERROR private trait `Foo` in public interface [E0445]
//~| NOTE can't leak private trait
pub fn foo<T: Foo> (t: T) {}
//~^ ERROR private trait `Foo` in public interface [E0445]
//~| NOTE can't leak private trait

fn main() {}
