#[no_mangle]
#[allow(improper_c_var_definitions)]
static FOO: () = ();

fn main() {
    extern "C" {
        fn FOO();
    }
    unsafe { FOO() } //~ ERROR: attempt to call an exported symbol that is not defined as a function
}
