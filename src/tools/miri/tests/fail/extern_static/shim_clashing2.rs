//@only-target: linux # we need a specific extern supported on this target

#[export_name = "environ"]
fn my_environ() {}
//~^ HELP: the `environ` symbol is defined here

fn main() {
    extern "C" {
        static environ: *const *const u8;
    }
    let _val = &raw const environ;
    //~^ ERROR: found `environ` symbol definition that clashes with a built-in shim
}
