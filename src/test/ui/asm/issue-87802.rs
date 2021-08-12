// only-x86_64
// Make sure rustc doesn't ICE on asm! when output type is !.

#![feature(asm)]

fn hmm() -> ! {
    let x;
    unsafe {
        asm!("/* {0} */", out(reg) x);
        //~^ ERROR cannot use value of type `!` for inline assembly
    }
    x
}

fn main() {
    hmm();
}
