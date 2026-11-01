//@ revisions: default feature
#![cfg_attr(feature, feature(arbitrary_self_types))]

use std::ops::Deref;
use std::marker::PhantomData;

struct Foo(u32);
impl Foo {
    fn get<R: Deref<Target = Self>>(self: R) -> u32 {
        //~^ ERROR: invalid generic `self` parameter type
        self.0
    }
    fn get1<R: Deref<Target = Self>>(self: &R) -> u32 {
        //~^ ERROR: invalid generic `self` parameter type
        self.0
    }
    fn get2<R: Deref<Target = Self>>(self: &mut R) -> u32 {
        //~^ ERROR: invalid generic `self` parameter type
        self.0
    }
    fn get3<R: Deref<Target = Self>>(self: std::rc::Rc<R>) -> u32 {
        //~^ ERROR: invalid generic `self` parameter type
        self.0
    }
    fn get4<R: Deref<Target = Self>>(self: &std::rc::Rc<R>) -> u32 {
        //~^ ERROR: invalid generic `self` parameter type
        self.0
    }
    fn get5<R: Deref<Target = Self>>(self: std::rc::Rc<&R>) -> u32 {
        //~^ ERROR: invalid generic `self` parameter type
        self.0
    }
    fn get6<FR: FindReceiver>(self: FR::Receiver, other: FR) -> u32 {
        //[default]~^ ERROR: `<FR as FindReceiver>::Receiver` cannot be used as the type of `self`
        42
    }
}


struct SmartPtr<'a, T: ?Sized>(&'a T);

impl<'a, T: ?Sized> Deref for SmartPtr<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unimplemented!()
    }
}

struct SmartPtr2<'a, T: ?Sized>(&'a T);

impl<'a, T: ?Sized> Deref for SmartPtr2<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unimplemented!()
    }
}

struct Bar<R>(std::marker::PhantomData<R>);

impl<R: std::ops::Deref<Target = Self>> Bar<R> {
    fn get(self: R) {}
    //[default]~^ ERROR: `R` cannot be used as the type of `self`
}

trait FindReceiver {
    type Receiver: Deref<Target = Foo>;
}

struct Silly;
impl FindReceiver for Silly {
    type Receiver = std::rc::Rc<Foo>;
}

fn main() {
    let mut foo = Foo(1);
    foo.get::<&Foo>();
    //[feature]~^ ERROR mismatched types
    foo.get::<std::rc::Rc<Foo>>();
    //[feature]~^ ERROR mismatched types

    let smart_ptr = SmartPtr(&foo);
    let smart_ptr2 = SmartPtr2(&foo);
    smart_ptr.get(); // this compiles
    smart_ptr.get::<SmartPtr2<Foo>>();
    //[feature]~^ ERROR mismatched types
    smart_ptr.get::<&Foo>();
    //[feature]~^ ERROR mismatched types

    let mut foo = Foo(1);
    // This test is slightly contrived in an attempt to generate a mismatched types
    // error for the self type below, without using the turbofish.
    foo.get6(Silly);
    //~^ ERROR type mismatch
    let mut foo = Foo(1);
    let foo = &foo;
    foo.get6(Silly);
    //~^ ERROR type mismatch

    let t = std::rc::Rc::new(Bar(std::marker::PhantomData));
    t.get();
    //~^ ERROR its trait bounds were not satisfied
    let t = &t;
    // This is a further attempt at triggering 'type mismatch' errors
    // from arbitrary self types without resorting to the turbofish.
    // Ideally, here, t is Thing<Rc<Target=Self>> while we're going to call
    // it with a &t method receiver. However, this doesn't work since that
    // type of t becomes recursive and trait bounds can't be satisfied.
    t.get();
    //~^ ERROR its trait bounds were not satisfied
}
