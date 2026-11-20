// This test checks that encapsulation symbols are not garbage collected by the linker.
// LLD will remove them by default, so this test checks that we pass `-znostart-stop-gc` to LLD
// to avoid that behavior. Without that flag, the test should fail.
// This test is inspired by the behavior of the linkme crate.
//
//@ build-pass
//@ only-x86_64-unknown-linux-gnu
//@ ignore-backends: gcc

unsafe extern "Rust" {
    // The __start_ section name is magical for the linker,
    // It will put link sections named EXTERNFNS after it.
    #[link_name = "__start_EXTERNFNS"]
    static SECTION_START: fn();
}

#[used]
#[unsafe(link_section = "EXTERNFNS")]
static EXTERN_FN_LOCAL: fn() = extern_fn;

fn extern_fn() {}

fn main() {
    // We need to reference the SECTION_START symbol to avoid it being garbage collected
    let slice = unsafe { SECTION_START };
}
