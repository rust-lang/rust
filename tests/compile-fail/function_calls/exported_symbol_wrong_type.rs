#[no_mangle]
static FOO: () = ();

fn main() {
    extern "C" {
        fn FOO();
    }
    unsafe { FOO() } //~ ERROR unsupported operation: can't call foreign function: FOO
}
