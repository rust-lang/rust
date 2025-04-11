//@ build-fail

pub const unsafe fn fake_type<T>() -> T {
    hint_unreachable() //~ NOTE_NONVIRAL inside
}

pub const unsafe fn hint_unreachable() -> ! {
    fake_type() //~ NOTE_NONVIRAL inside
}

trait Const {
    const CONSTANT: i32 = unsafe { fake_type() }; //~ ERROR evaluation of `fake_type::<!>` failed
}

impl<T> Const for T {}

pub fn main() -> () {
    dbg!(i32::CONSTANT);
}
