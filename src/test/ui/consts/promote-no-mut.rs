// ignore-tidy-linelength
// We do not promote mutable references.
static mut TEST1: Option<&mut [i32]> = Some(&mut [1, 2, 3]); //~ ERROR temporary value dropped while borrowed

static mut TEST2: &'static mut [i32] = {
    let x = &mut [1,2,3]; //~ ERROR temporary value dropped while borrowed
    x
};

fn main() {}
