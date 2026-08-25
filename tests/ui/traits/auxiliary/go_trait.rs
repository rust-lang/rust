#![feature(specialization)]

// Common code used for tests that model the Fn/FnMut/FnOnce hierarchy.

pub trait Go {
    fn go(&self, arg: isize);
}

pub fn go<G:Go>(this: &G, arg: isize) {
    this.go(arg)
}

pub trait GoMut {
    fn go_mut(&mut self, arg: isize);
}

pub fn go_mut<G:GoMut>(this: &mut G, arg: isize) {
    this.go_mut(arg)
}

pub trait GoOnce {
    fn go_once(self, arg: isize);
}

pub fn go_once<G:GoOnce>(this: G, arg: isize) {
    this.go_once(arg)
}

impl<G> GoMut for G
    where G : Go
{
    default fn go_mut(&mut self, arg: isize) {
        go(&*self, arg)
    }
}

impl<G> GoOnce for G
    where G : GoMut
{
    default fn go_once(mut self, arg: isize) {
        go_mut(&mut self, arg)
    }
}
