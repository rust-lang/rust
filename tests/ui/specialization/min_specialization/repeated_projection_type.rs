// Test that projection bounds can't be specialized on.

#![feature(min_specialization)]

trait X {
    fn f();
}
trait Id {
    type This;
}
impl<T> Id for T {
    type This = T;
}

impl<T: Id> X for T {
    default fn f() {}
}

impl<I, V: Id<This = (I,)>> X for V {
    //~^ ERROR cannot specialize on
    fn f() {}
}

fn main() {}
