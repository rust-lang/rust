fn foo1<T:Copy<U>, U>(x: T) {}
//~^ ERROR this trait takes 0 type arguments but 1 type argument was supplied

trait Trait: Copy<dyn Send> {}
//~^ ERROR this trait takes 0 type arguments but 1 type argument was supplied

struct MyStruct1<T: Copy<T>>;
//~^ ERROR this trait takes 0 type arguments but 1 type argument was supplied

struct MyStruct2<'a, T: Copy<'a>>;
//~^ ERROR this trait takes 0 lifetime arguments but 1 lifetime argument was supplied

fn foo2<'a, T:Copy<'a, U>, U>(x: T) {}
//~^ ERROR this trait takes 0 lifetime arguments but 1 lifetime argument was supplied
//~| ERROR this trait takes 0 type arguments but 1 type argument was supplied

fn main() { }
