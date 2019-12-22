// build-fail

pub const unsafe fn fake_type<T>() -> T {
    hint_unreachable()
}

pub const unsafe fn hint_unreachable() -> ! {
    fake_type() //~ ERROR cycle detected when const-evaluating `hint_unreachable` [E0391]
}

trait Const {
    const CONSTANT: i32 = unsafe { fake_type() };
}

impl <T> Const for T {}

pub fn main() -> () {
    dbg!(i32::CONSTANT);
}
