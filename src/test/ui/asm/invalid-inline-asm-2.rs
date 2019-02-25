#![feature(asm)]

fn main() {
    let a: usize;

    unsafe {
        asm!("" : "=d"(a) : : : );
        //~^ ERROR couldn't allocate output register for constraint 'd'
    }
}
