//@ check-fail
//@ known-bug: #103705

// The output of this currently suggests writing a closure in the qualified path.

trait MyTrait<T> {
   fn lol<F:FnOnce()>(&self, f:F) -> u16;
}

struct Qqq;

impl MyTrait<u32> for Qqq{
   fn lol<F:FnOnce()>(&self, _f:F) -> u16 { 5 }
}
impl MyTrait<u64> for Qqq{
   fn lol<F:FnOnce()>(&self, _f:F) -> u16 { 6 }
}

fn main() {
    let q = Qqq;
    q.lol(||());
}
