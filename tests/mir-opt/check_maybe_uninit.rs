use std::mem::MaybeUninit;

// EMIT_MIR check_maybe_uninit.main.CheckMaybeUninit.diff
fn main() {
    unsafe {
        let _ = MaybeUninit::<u8>::uninit().assume_init();
        let _ = MaybeUninit::<String>::uninit().assume_init();
    }
}
