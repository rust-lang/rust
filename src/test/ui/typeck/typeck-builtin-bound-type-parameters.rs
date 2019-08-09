fn foo1<T:Copy<U>, U>(x: T) {}
//~^ ERROR wrong number of type arguments: expected 0, found 1 [E0107]

trait Trait: Copy<dyn Send> {}
//~^ ERROR wrong number of type arguments: expected 0, found 1 [E0107]

struct MyStruct1<T: Copy<T>>;
//~^ ERROR wrong number of type arguments: expected 0, found 1 [E0107]

struct MyStruct2<'a, T: Copy<'a>>;
//~^ ERROR: wrong number of lifetime arguments: expected 0, found 1 [E0107]


fn foo2<'a, T:Copy<'a, U>, U>(x: T) {}
//~^ ERROR wrong number of type arguments: expected 0, found 1 [E0107]
//~| ERROR: wrong number of lifetime arguments: expected 0, found 1

fn main() {
}
