// check-pass
#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

trait Specializable { type Output; }

impl<T> Specializable for T {
    default type Output = u16;
}

fn main() {
    unsafe {
        std::mem::transmute::<u16, <() as Specializable>::Output>(0);
        //~^ WARN relying on the underlying type of an opaque type in the type system
        //~| WARN this was previously accepted by the compiler but is being phased out
    }
}
