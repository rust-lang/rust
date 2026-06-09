extern "C" {
    fn printf(_: *const u8, ...) -> u32;
}

fn main() {
    unsafe { printf(); }
    //~^ ERROR E0060
}
