//@ build-fail

pub const unsafe fn fake_type<T>() -> T {
    hint_unreachable() //~ ERROR evaluation of `<i32 as Const>::CONSTANT` failed
}

pub const unsafe fn hint_unreachable() -> ! {
    fake_type()
}

trait Const {
    const CONSTANT: i32 = unsafe { fake_type() };
}

impl<T> Const for T {}

pub fn main() -> () {
    dbg!(i32::CONSTANT); //~ constant
}
