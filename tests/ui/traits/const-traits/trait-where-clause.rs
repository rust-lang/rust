#![feature(const_trait_impl)]

#[const_trait]
trait Bar {}

trait Foo {
    fn a();
    fn b() where Self: [const] Bar;
    //~^ ERROR `[const]` is not allowed here
    fn c<T: [const] Bar>();
    //~^ ERROR `[const]` is not allowed here
}

fn test1<T: Foo>() {
    T::a();
    T::b();
    //~^ ERROR the trait bound
    T::c::<T>();
    //~^ ERROR the trait bound
}

fn test2<T: Foo + Bar>() {
    T::a();
    T::b();
    T::c::<T>();
}

fn main() {}
