#![crate_type = "lib"]

#![feature(box_syntax)]

static mut COUNT: u64 = 1;

pub fn get_count() -> u64 { unsafe { COUNT } }

#[derive(Copy, Clone)]
pub struct Foo;

impl Foo {
    pub fn run_trait(self) {
        unsafe { COUNT *= 17; }
        // Test internal call.
        Bar::foo1(&self);
        Bar::foo2(self);
        Bar::foo3(box self);

        Bar::bar1(&self);
        Bar::bar2(self);
        Bar::bar3(box self);
    }
}

pub trait Bar : Sized {
    fn foo1(&self);
    fn foo2(self);
    fn foo3(self: Box<Self>);

    fn bar1(&self) {
        unsafe { COUNT *= 7; }
    }
    fn bar2(self) {
        unsafe { COUNT *= 11; }
    }
    fn bar3(self: Box<Self>) {
        unsafe { COUNT *= 13; }
    }
}

impl Bar for Foo {
    fn foo1(&self) {
        unsafe { COUNT *= 2; }
    }

    fn foo2(self) {
        unsafe { COUNT *= 3; }
    }

    fn foo3(self: Box<Foo>) {
        unsafe { COUNT *= 5; }
    }
}
