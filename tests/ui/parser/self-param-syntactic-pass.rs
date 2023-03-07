// This test ensures that `self` is syntactically accepted in all places an `FnDecl` is parsed.
// FIXME(Centril): For now closures are an exception.

// check-pass

fn main() {}

#[cfg(FALSE)]
fn free() {
    fn f(self) {}
    fn f(mut self) {}
    fn f(&self) {}
    fn f(&mut self) {}
    fn f(&'a self) {}
    fn f(&'a mut self) {}
    fn f(self: u8) {}
    fn f(mut self: u8) {}
}

#[cfg(FALSE)]
extern "C" {
    fn f(self);
    fn f(mut self);
    fn f(&self);
    fn f(&mut self);
    fn f(&'a self);
    fn f(&'a mut self);
    fn f(self: u8);
    fn f(mut self: u8);
}

#[cfg(FALSE)]
trait X {
    fn f(self) {}
    fn f(mut self) {}
    fn f(&self) {}
    fn f(&mut self) {}
    fn f(&'a self) {}
    fn f(&'a mut self) {}
    fn f(self: u8) {}
    fn f(mut self: u8) {}
}

#[cfg(FALSE)]
impl X for Y {
    fn f(self) {}
    fn f(mut self) {}
    fn f(&self) {}
    fn f(&mut self) {}
    fn f(&'a self) {}
    fn f(&'a mut self) {}
    fn f(self: u8) {}
    fn f(mut self: u8) {}
}

#[cfg(FALSE)]
impl X for Y {
    type X = fn(self);
    type X = fn(mut self);
    type X = fn(&self);
    type X = fn(&mut self);
    type X = fn(&'a self);
    type X = fn(&'a mut self);
    type X = fn(self: u8);
    type X = fn(mut self: u8);
}
