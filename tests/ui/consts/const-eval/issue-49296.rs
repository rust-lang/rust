// issue-49296: Unsafe shenigans in constants can result in missing errors

use std::mem::transmute;

const fn wat(x: u64) -> &'static u64 {
    unsafe { transmute(&x) }
}

const X: u64 = *wat(42);
//~^ ERROR dangling

fn main() {
    println!("{}", X);
}
