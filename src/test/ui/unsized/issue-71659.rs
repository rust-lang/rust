#![feature(unsize)]

use std::marker::Unsize;
use std::rc::Rc;
use std::sync::Arc;

pub trait CastTo<T: ?Sized>: Unsize<T> {
    fn cast_to(&self) -> &T;
    fn cast_mut_to(&mut self) -> &mut T;
    fn into_cast_to(self: Box<Self>) -> Box<T>;
    fn cast_rc_to(self: Rc<Self>) -> Rc<T>;
    fn cast_arc_to(self: Arc<Self>) -> Arc<T>;
}

impl<T: ?Sized> Cast for T {}
pub trait Cast {
    fn cast<T: ?Sized>(&self) -> &T
    where
        Self: CastTo<T>,
    {
        self
    }

    fn cast_mut<T>(&mut self) -> &mut T
    where
        Self: CastTo<T>,
    {
        self.cast_mut_to()
    }

    fn into_cast<T>(self: Box<Self>) -> Box<T>
    where
        Self: CastTo<T>,
    {
        self.into_cast_to()
    }

    fn cast_rc<T>(self: Rc<Self>) -> Rc<T>
    where
        Self: CastTo<T>,
    {
        self.cast_rc_to()
    }

    fn cast_arc<T>(self: Arc<Self>) -> Arc<T>
    where
        Self: CastTo<T>,
    {
        self.cast_arc_to()
    }
}
impl<T: ?Sized, U: ?Sized + Unsize<T>> CastTo<T> for U {
    fn cast_to(&self) -> &T {
        self
    }

    fn cast_mut_to(&mut self) -> &mut T {
        self
    }

    fn into_cast_to(self: Box<Self>) -> Box<T> {
        self
    }

    fn cast_rc_to(self: Rc<Self>) -> Rc<T> {
        self
    }

    fn cast_arc_to(self: Arc<Self>) -> Arc<T> {
        self
    }
}

pub trait Foo {
    fn foo(&self) {
        println!("Foo({})", core::any::type_name::<Self>());
    }
}

pub trait Bar: CastTo<dyn Foo> + CastTo<dyn core::fmt::Debug> + CastTo<[i32]> {
    fn bar(&self) {
        println!("Bar({})", core::any::type_name::<Self>());
    }
}

impl Foo for [i32; 10] {}
impl Bar for [i32; 10] {}

fn main() {
    let x = [0; 10];
    let x: Box<dyn Bar> = Box::new(x);
    let x = (*x).cast::<[i32]>();
    //~^ ERROR: the trait bound `dyn Bar: CastTo<[i32]>` is not satisfied
}
