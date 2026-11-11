//@only-target: linux # we need a specific extern supported on this target

extern "C" {
    static mut environ: *const *const u8;
}

#[export_name = "environ"]
static mut MY_ENVIRON: *const *const u8 = std::ptr::null();
//~^ HELP: the `environ` symbol is defined here

fn main() {
    let _val = &raw const MY_ENVIRON;
    let _val = &raw const environ;
    //~^ ERROR: found `environ` symbol definition that clashes with a built-in shim
}
