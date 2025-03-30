use std::mem::MaybeUninit;

fn main() {
    let uninit: MaybeUninit<i32> = MaybeUninit::uninit();
    let bad_ref: &i32 = unsafe { uninit.assume_init_ref() };
    let &(0 | _) = bad_ref;
    //~^ ERROR: Undefined Behavior: using uninitialized data, but this operation requires initialized memory
}
