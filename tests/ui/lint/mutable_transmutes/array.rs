use std::mem::transmute;

fn main() {
    let _a: [&mut u8; 2] = unsafe { transmute([&1u8; 2]) };
    //~^ ERROR transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell
}
