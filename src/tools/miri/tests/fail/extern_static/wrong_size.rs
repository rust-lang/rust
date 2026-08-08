#[no_mangle]
static FOO: u8 = 42;

fn main() {
    extern "Rust" {
        static FOO: u16;
    }
    let _val = unsafe { (&raw const FOO).read() };
    //~^ ERROR: extern static `FOO` has been declared as `wrong_size::main::FOO` with a size of 2 bytes
}
