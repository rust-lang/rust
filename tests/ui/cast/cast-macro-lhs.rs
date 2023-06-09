// Test to make sure we suggest "consider casting" on the right span

macro_rules! foo {
    () => { 0 }
}

fn main() {
    let x = foo!() as *const [u8];
    //~^ ERROR cannot cast `usize` to a pointer that is wide
    //~| NOTE creating a `*const [u8]` requires both an address and a length
    //~| NOTE consider casting this expression to `*const ()`, then using `core::ptr::from_raw_parts`
}
