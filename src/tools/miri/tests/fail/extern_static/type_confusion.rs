#[no_mangle]
static FOO: u8 = 42;

fn main() {
    extern "Rust" {
        static FOO: bool;
    }
    // Type confusion between u8 (value 42) and bool: reading as bool is UB
    // because 42 is not a valid boolean value (must be 0 or 1).
    unsafe {
        (&raw const FOO).read();
        //~^ ERROR: /constructing invalid value of type bool/
    }
}
