// Tests that transmuting from &T to &mut T is Undefined Behavior.

use std::mem::transmute;

fn main() {
    let _a: &mut u8 = unsafe { transmute(&1u8) };
    //~^ ERROR transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell
}
