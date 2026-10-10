// Check that you can't dereference invalid raw pointers in constants.

fn main() {
    static C: u64 = unsafe { *(0xdeadbeef as *const u64) };
    //~^ ERROR dangling pointer
    println!("{}", C);
}
