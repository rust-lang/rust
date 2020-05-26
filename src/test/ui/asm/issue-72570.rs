// only-x86_64

#![feature(asm)]

fn main() {
    unsafe {
        asm!("", in("invalid") "".len());
        //~^ ERROR: invalid register `invalid`: unknown register
    }
}
