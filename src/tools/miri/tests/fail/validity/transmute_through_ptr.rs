#[repr(u32)]
#[derive(Debug)]
enum Bool {
    True,
}

fn evil(x: &mut Bool) {
    let x = x as *mut _ as *mut u32;
    unsafe { *x = 44 }; // out-of-bounds enum tag
}

#[rustfmt::skip] // rustfmt bug: https://github.com/rust-lang/rustfmt/issues/5391
fn main() {
    let mut x = Bool::True;
    evil(&mut x);
    let y = x; // reading this ought to be enough to trigger validation
    //~^ ERROR: constructing invalid value at .<enum-tag>: encountered 0x0000002c, but expected a valid enum tag
    println!("{:?}", y); // make sure it is used (and not optimized away)
}
