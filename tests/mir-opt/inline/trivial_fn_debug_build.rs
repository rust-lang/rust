// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// compile-flags: -Copt-level=0 -Zmir-opt-level=1
#![crate_type = "lib"]

// Test that we still inline trivial functions even in a debug build

pub struct Thing {
    inner: u8,
}

impl Thing {
    #[inline]
    pub fn get(&self) -> u8 {
        self.inner
    }
}

// EMIT_MIR trivial_fn_debug_build.wrapper.Inline.diff
// EMIT_MIR trivial_fn_debug_build.wrapper.PreCodegen.after.mir
pub fn wrapper(t: &Thing) -> u8 {
    t.get()
}
