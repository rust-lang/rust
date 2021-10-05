// Check that you can't dereference invalid raw pointers in constants.

fn main() {
    static C: u64 = unsafe {*(0xdeadbeef as *const u64)};
    //~^ ERROR could not evaluate static initializer
    println!("{}", C);
}
