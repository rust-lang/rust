#![warn(clippy::iter_not_returning_iterator)]

struct Data {
    begin: u32,
}

struct Counter {
    count: u32,
}

impl Data {
    fn iter(&self) -> Counter {
        todo!()
    }

    fn iter_mut(&self) -> Counter {
        todo!()
    }
}

struct Data2 {
    begin: u32,
}

struct Counter2 {
    count: u32,
}

impl Data2 {
    fn iter(&self) -> Counter2 {
        //~^ ERROR: this method is named `iter` but its return type does not implement `Iterat
        //~| NOTE: `-D clippy::iter-not-returning-iterator` implied by `-D warnings`
        todo!()
    }

    fn iter_mut(&self) -> Counter2 {
        //~^ ERROR: this method is named `iter_mut` but its return type does not implement `It
        todo!()
    }
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

// Issue #8225
trait Iter {
    type I;
    fn iter(&self) -> Self::I;
    //~^ ERROR: this method is named `iter` but its return type does not implement `Iterat
}

impl Iter for () {
    type I = core::slice::Iter<'static, ()>;
    fn iter(&self) -> Self::I {
        [].iter()
    }
}

struct S;
impl S {
    fn iter(&self) -> <() as Iter>::I {
        ().iter()
    }
}

struct S2([u8]);
impl S2 {
    fn iter(&self) -> core::slice::Iter<u8> {
        self.0.iter()
    }
}

fn main() {}
