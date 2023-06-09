trait T0 {}
trait T1: T0 {}

trait T2 {}

impl<'a> T0 for &'a (dyn T2 + 'static) {}

impl T1 for &dyn T2 {}
//~^ ERROR mismatched types

fn main() {}
