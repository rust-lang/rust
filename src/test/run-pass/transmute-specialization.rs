#![feature(specialization)]

trait Specializable { type Output; }

impl<T> Specializable for T {
    default type Output = u16;
}

fn main() {
    unsafe {
        std::mem::transmute::<u16, <() as Specializable>::Output>(0);
    }
}
