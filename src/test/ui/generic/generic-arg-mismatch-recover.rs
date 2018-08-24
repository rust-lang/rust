struct Foo<'a, T: 'a>(&'a T);

struct Bar<'a>(&'a ());

fn main() {
    Foo::<'static, 'static, ()>(&0); //~ ERROR wrong number of lifetime arguments
    //~^ ERROR mismatched types

    Bar::<'static, 'static, ()>(&()); //~ ERROR wrong number of lifetime arguments
    //~^ ERROR wrong number of type arguments
}
