#![feature(asm)]

fn main() {
    unsafe {
        asm!("", options(may_unwind));
        //~^ ERROR the `may_unwind` option is unstable
    }
}
