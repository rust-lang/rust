use std::mem::transmute;

#[cfg(target_pointer_width = "32")]
type TwoPtrs = i64;
#[cfg(target_pointer_width = "64")]
type TwoPtrs = i128;

fn main() {
    for &my_bool in &[true, false] {
        let mask = -(my_bool as TwoPtrs); // false -> 0, true -> -1 aka !0
        // This is branchless code to select one or the other pointer.
        // For now, Miri brafs on it, but if this code ever passes we better make sure it behaves correctly.
        let val = unsafe {
            transmute::<_, &str>(
                !mask & transmute::<_, TwoPtrs>("false !") | mask & transmute::<_, TwoPtrs>("true !"), //~ERROR encountered (potentially part of) a pointer, but expected plain (non-pointer) bytes
            )
        };
        println!("{}", val);
    }
}
