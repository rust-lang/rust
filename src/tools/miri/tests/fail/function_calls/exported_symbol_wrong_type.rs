#[no_mangle]
static FOO: () = ();

fn main() {
    extern "C" {
        fn FOO();
    }
    unsafe { FOO() } //~ ERROR: attempt to call an exported symbol that is not defined as a function
}
