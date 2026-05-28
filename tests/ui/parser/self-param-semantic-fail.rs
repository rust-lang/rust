// This test ensures that `self` is semantically rejected
// in contexts with `FnDecl` but outside of associated `fn`s.
// FIXME(Centril): For now closures are an exception.

fn main() {}

fn free() {
    fn f1(self) {}
    //~^ ERROR `self` parameter is only allowed in associated functions
    fn f2(mut self) {}
    //~^ ERROR `self` parameter is only allowed in associated functions
    fn f3(&self) {}
    //~^ ERROR `self` parameter is only allowed in associated functions
    fn f4(&mut self) {}
    //~^ ERROR `self` parameter is only allowed in associated functions
    fn f5<'a>(&'a self) {}
    //~^ ERROR `self` parameter is only allowed in associated functions
    fn f6<'a>(&'a mut self) {}
    //~^ ERROR `self` parameter is only allowed in associated functions
    fn f7(self: u8) {}
    //~^ ERROR `self` parameter is only allowed in associated functions
    fn f8(mut self: u8) {}
    //~^ ERROR `self` parameter is only allowed in associated functions
}

extern "C" {
    fn f1(self);
    //~^ ERROR `self` parameter is only allowed in associated functions
    fn f2(mut self);
    //~^ ERROR `self` parameter is only allowed in associated functions
    //~| ERROR patterns aren't allowed in
    fn f3(&self);
    //~^ ERROR `self` parameter is only allowed in associated functions
    fn f4(&mut self);
    //~^ ERROR `self` parameter is only allowed in associated functions
    fn f5<'a>(&'a self);
    //~^ ERROR `self` parameter is only allowed in associated functions
    fn f6<'a>(&'a mut self);
    //~^ ERROR `self` parameter is only allowed in associated functions
    fn f7(self: u8);
    //~^ ERROR `self` parameter is only allowed in associated functions
    fn f8(mut self: u8);
//~^ ERROR `self` parameter is only allowed in associated functions
//~| ERROR patterns aren't allowed in
}

type X1 = fn(self);
//~^ ERROR `self` parameter is only allowed in associated functions
type X2 = fn(mut self);
//~^ ERROR `self` parameter is only allowed in associated functions
//~| ERROR patterns aren't allowed in
type X3 = fn(&self);
//~^ ERROR `self` parameter is only allowed in associated functions
type X4 = fn(&mut self);
//~^ ERROR `self` parameter is only allowed in associated functions
type X5 = for<'a> fn(&'a self);
//~^ ERROR `self` parameter is only allowed in associated functions
type X6 = for<'a> fn(&'a mut self);
//~^ ERROR `self` parameter is only allowed in associated functions
type X7 = fn(self: u8);
//~^ ERROR `self` parameter is only allowed in associated functions
type X8 = fn(mut self: u8);
//~^ ERROR `self` parameter is only allowed in associated functions
//~| ERROR patterns aren't allowed in
