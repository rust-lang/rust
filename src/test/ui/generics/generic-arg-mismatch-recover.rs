struct Foo<'a, T: 'a>(&'a T);

struct Bar<'a>(&'a ());

fn main() {
    Foo::<'static, 'static, ()>(&0);
    //~^ ERROR this struct takes 1 lifetime argument but 2 lifetime arguments were supplied

    Bar::<'static, 'static, ()>(&());
    //~^ ERROR this struct takes 1 lifetime argument but 2 lifetime arguments were supplied
    //~| ERROR this struct takes 0 type arguments but 1 type argument was supplied
}
