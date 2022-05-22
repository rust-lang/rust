#![feature(const_trait_impl)]

trait Bar {}

trait Foo {
    fn a();
    fn b() where Self: ~const Bar;
    fn c<T: ~const Bar>();
}

const fn test1<T: ~const Foo + Bar>() {
    T::a();
    T::b();
    //~^ ERROR the trait bound
    T::c::<T>();
    //~^ ERROR the trait bound
}

const fn test2<T: ~const Foo + ~const Bar>() {
    T::a();
    T::b();
    T::c::<T>();
}

fn test3<T: Foo>() {
    T::a();
    T::b();
    //~^ ERROR the trait bound
    T::c::<T>();
    //~^ ERROR the trait bound
}

fn test4<T: Foo + Bar>() {
    T::a();
    T::b();
    T::c::<T>();
}

fn main() {}
