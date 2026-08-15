struct Foo<'a, T: 'a>(&'a T);

struct Bar<'a>(&'a ());

fn main() {
    Foo::<'static, 'static, ()>(&0);
    //~^ ERROR struct takes 1 lifetime argument but 2 lifetime arguments were supplied

    Bar::<'static, 'static, ()>(&());
    //~^ ERROR struct takes 1 lifetime argument but 2 lifetime arguments were supplied
    //~| ERROR struct takes 0
}
