// build-fail

pub const unsafe fn fake_type<T>() -> T {
    hint_unreachable() //~ ERROR any use of this value will cause an error [const_err]
}

pub const unsafe fn hint_unreachable() -> ! {
    fake_type()
}

trait Const {
    const CONSTANT: i32 = unsafe { fake_type() };
}

impl<T> Const for T {}

pub fn main() -> () {
    dbg!(i32::CONSTANT); //~ ERROR erroneous constant used
}
