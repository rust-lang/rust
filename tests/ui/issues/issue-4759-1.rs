//@ run-pass
trait U { fn f(self); }
impl U for isize { fn f(self) {} }
pub fn main() { 4.f(); }
