use core::process;

#[test]
fn const_breakpoint() {
    // Ensure this continues in CTFE
    const { process::resumable_breakpoint() };

    // Miri should also continue, while native runtime will likely abort
    if cfg!(miri) {
        process::resumable_breakpoint();
    }
}
