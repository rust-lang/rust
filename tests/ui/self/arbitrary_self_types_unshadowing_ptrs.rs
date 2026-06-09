#![feature(arbitrary_self_types_pointers)]
#![feature(arbitrary_self_types)]

pub struct A;

// The receiver of the potentially shadowed method
// precisely matches that of the shadower
impl A {
    pub fn f(self: Wrapper<Self>) -> i32 { 1 }
    pub fn g(self: &Wrapper<Self>) -> i32 { 2 }
    pub fn h(self: &mut Wrapper<Self>) -> i32 { 3 }
    pub fn i(self: *const Wrapper<Self>) -> i32 { 4 }
}

// The receiver of the potentially shadowed method is a reference
pub struct B;

impl B {
    pub fn f(self: &Wrapper<Self>) -> i32 { 9 }
}

// The receiver of the potentially shadowed method is a mut reference

pub struct C;

impl C {
    pub fn f(self: &mut Wrapper<Self>) -> i32 { 10 }
    pub fn g(self: &mut Wrapper<Self>) -> i32 { 11 }
}

pub struct Wrapper<T>(T);

impl<T> core::ops::Receiver for Wrapper<T> {
    type Target = T;
}

impl<T> Wrapper<T> {
    pub fn f(self) -> i32 { 5 }
    pub fn g(&self) -> i32 { 6 }
    pub fn h(&mut self) -> i32 { 7 }
    pub fn i(self: *const Self) -> i32 { 8 }
}

fn main() {
    assert_eq!(Wrapper(A).f(), 1);
    //~^ ERROR: multiple applicable items in scope
    assert_eq!(Wrapper(A).g(), 2);
    //~^ ERROR: multiple applicable items in scope
    assert_eq!(Wrapper(A).h(), 3);
    //~^ ERROR: multiple applicable items in scope
    let a = Wrapper(A);
    let a_ptr = &a as *const Wrapper<A>;
    assert_eq!(a_ptr.i(), 4);
    //~^ ERROR: multiple applicable items in scope
    assert_eq!(Wrapper(B).f(), 9);
    //~^ ERROR: multiple applicable items in scope
    assert_eq!(Wrapper(C).f(), 10);
    //~^ ERROR: multiple applicable items in scope
    assert_eq!(Wrapper(C).g(), 11);
    //~^ ERROR: multiple applicable items in scope
}
