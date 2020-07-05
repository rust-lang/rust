#[repr(u32)]
enum Bool { True }

fn evil(x: &mut Bool) {
    let x = x as *mut _ as *mut u32;
    unsafe { *x = 44; } // out-of-bounds enum tag
}

fn main() {
    let mut x = Bool::True;
    evil(&mut x);
    let _y = x; // reading this ought to be enough to trigger validation
    //~^ ERROR encountered 0x0000002c at .<enum-tag>, but expected a valid enum tag
}
