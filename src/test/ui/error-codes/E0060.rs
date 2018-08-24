extern "C" {
    fn printf(_: *const u8, ...) -> u32;
}

fn main() {
    unsafe { printf(); }
    //~^ ERROR E0060
    //~| expected at least 1 parameter
}
