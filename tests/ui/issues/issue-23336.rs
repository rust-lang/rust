//@ run-pass
pub trait Data { fn doit(&self) {} }
impl<T> Data for T {}
pub trait UnaryLogic { type D: Data; }
impl UnaryLogic for () { type D = i32; }

pub fn crashes<T: UnaryLogic>(t: T::D) {
    t.doit();
}

fn main() { crashes::<()>(0); }
