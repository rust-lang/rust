// This test ensures that `self` is semantically rejected
// in contexts with `FnDecl` but outside of associated `fn`s.
// FIXME(Centril): For now closures are an exception.

fn main() {}

fn free() {
    fn f1(self) {}
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    fn f2(mut self) {}
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    fn f3(&self) {}
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    fn f4(&mut self) {}
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    fn f5<'a>(&'a self) {}
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    fn f6<'a>(&'a mut self) {}
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    fn f7(self: u8) {}
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    fn f8(mut self: u8) {}
    //~^ ERROR `self` parameter only allowed in associated `fn`s
}

extern {
    fn f1(self);
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    fn f2(mut self);
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    //~| ERROR patterns aren't allowed in
    fn f3(&self);
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    fn f4(&mut self);
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    fn f5<'a>(&'a self);
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    fn f6<'a>(&'a mut self);
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    fn f7(self: u8);
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    fn f8(mut self: u8);
    //~^ ERROR `self` parameter only allowed in associated `fn`s
    //~| ERROR patterns aren't allowed in
}

type X1 = fn(self);
//~^ ERROR `self` parameter only allowed in associated `fn`s
type X2 = fn(mut self);
//~^ ERROR `self` parameter only allowed in associated `fn`s
//~| ERROR patterns aren't allowed in
type X3 = fn(&self);
//~^ ERROR `self` parameter only allowed in associated `fn`s
type X4 = fn(&mut self);
//~^ ERROR `self` parameter only allowed in associated `fn`s
type X5 = for<'a> fn(&'a self);
//~^ ERROR `self` parameter only allowed in associated `fn`s
type X6 = for<'a> fn(&'a mut self);
//~^ ERROR `self` parameter only allowed in associated `fn`s
type X7 = fn(self: u8);
//~^ ERROR `self` parameter only allowed in associated `fn`s
type X8 = fn(mut self: u8);
//~^ ERROR `self` parameter only allowed in associated `fn`s
//~| ERROR patterns aren't allowed in
