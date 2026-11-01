//@ build-fail
//@ dont-require-annotations: NOTE

pub const unsafe fn fake_type<T>() -> T {
    hint_unreachable() //~ NOTE inside
}

pub const unsafe fn hint_unreachable() -> ! {
    fake_type() //~ NOTE inside
}

trait Const {
    const CONSTANT: i32 = unsafe { fake_type() }; //~ ERROR reached the configured maximum number of stack frames
    //~^ NOTE evaluation of `<i32 as Const>::CONSTANT` failed inside this call
}

impl<T> Const for T {}

pub fn main() -> () {
    dbg!(i32::CONSTANT);
}
