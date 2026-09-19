#[allow(non_snake_case)]
#[no_mangle]
fn FOO() {}

fn main() {
    extern "Rust" {
        static FOO: ();
    }
    let _val = &raw const FOO;
    //~^ ERROR: attempt to access an exported symbol `FOO` that is not defined as a static
}
