// only-x86_64

use std::arch::asm;

fn main() {
    unsafe {
        asm!("", options(may_unwind));
        //~^ ERROR the `may_unwind` option is unstable
    }
}
