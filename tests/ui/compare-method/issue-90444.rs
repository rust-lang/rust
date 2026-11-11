pub struct A;
impl From<fn((), (), &())> for A {
    fn from(_: fn((), (), &mut ())) -> Self {
        //~^ error: method `from` has an incompatible type for trait
        loop {}
    }
}

pub struct B;
impl From<fn((), (), u32)> for B {
    fn from(_: fn((), (), u64)) -> Self {
        //~^ error: method `from` has an incompatible type for trait
        loop {}
    }
}

fn main() {}
