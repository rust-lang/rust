// Check that you can't dereference raw pointers in constants.

fn main() {
    static C: u64 = unsafe {*(0xdeadbeef as *const u64)};
    //~^ ERROR dereferencing raw pointers in statics is unstable
    println!("{}", C);
}
