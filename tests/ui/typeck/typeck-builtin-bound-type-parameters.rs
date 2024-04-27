fn foo1<T:Copy<U>, U>(x: T) {}
//~^ ERROR trait takes 0 generic arguments but 1 generic argument was supplied

trait Trait: Copy<dyn Send> {}
//~^ ERROR trait takes 0 generic arguments but 1 generic argument was supplied
//~| ERROR trait takes 0 generic arguments but 1 generic argument was supplied
//~| ERROR trait takes 0 generic arguments but 1 generic argument was supplied

struct MyStruct1<T: Copy<T>>(T);
//~^ ERROR trait takes 0 generic arguments but 1 generic argument was supplied

struct MyStruct2<'a, T: Copy<'a>>(&'a T);
//~^ ERROR trait takes 0 lifetime arguments but 1 lifetime argument was supplied

fn foo2<'a, T:Copy<'a, U>, U>(x: T) {}
//~^ ERROR trait takes 0 lifetime arguments but 1 lifetime argument was supplied
//~| ERROR trait takes 0 generic arguments but 1 generic argument was supplied

fn main() { }
