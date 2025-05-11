//@ build-fail
//@ dont-require-annotations: NOTE

pub const unsafe fn fake_type<T>() -> T {
    hint_unreachable() //~ NOTE inside
}

pub const unsafe fn hint_unreachable() -> ! {
    fake_type() //~ NOTE inside
}

trait Const {
    const CONSTANT: i32 = unsafe { fake_type() }; //~ ERROR evaluation of `fake_type::<!>` failed
}

impl<T> Const for T {}

pub fn main() -> () {
    dbg!(i32::CONSTANT);
}
