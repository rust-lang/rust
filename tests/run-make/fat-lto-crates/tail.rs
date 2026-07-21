// Keep this crate outside the core to exercise an always-inline ThinLTO import
// whose body changes without changing the merged core.

#[inline(always)]
#[unsafe(no_mangle)]
pub extern "C" fn finish(x: u32) -> u32 {
    #[cfg(not(tail_b))]
    {
        x.wrapping_mul(3).wrapping_add(7)
    }
    #[cfg(tail_b)]
    {
        x.wrapping_mul(5).wrapping_add(11)
    }
}
