fn foo1<T:Copy<U>, U>(x: T) {}
//~^ ERROR wrong number of type arguments: expected 0, found 1 [E0244]

trait Trait: Copy<Send> {}
//~^ ERROR wrong number of type arguments: expected 0, found 1 [E0244]

struct MyStruct1<T: Copy<T>>;
//~^ ERROR wrong number of type arguments: expected 0, found 1 [E0244]

struct MyStruct2<'a, T: Copy<'a>>;
//~^ ERROR: wrong number of lifetime arguments: expected 0, found 1


fn foo2<'a, T:Copy<'a, U>, U>(x: T) {}
//~^ ERROR wrong number of type arguments: expected 0, found 1 [E0244]
//~| ERROR: wrong number of lifetime arguments: expected 0, found 1

fn main() {
}
