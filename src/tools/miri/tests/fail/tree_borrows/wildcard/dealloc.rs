//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks that deallocation through a wildcard ref fails,
/// if all exposed references are disabled.
pub fn main() {
    use std::alloc::Layout;
    let x = unsafe { std::alloc::alloc_zeroed(Layout::new::<u32>()) as *mut u32 };

    let ref1 = unsafe { &mut *x };
    let ref2 = unsafe { &mut *x };

    let int = ref1 as *mut u32 as usize;
    let wild = int as *mut u32;
    // Disables ref1 and therefore also wild.
    *ref2 = 14;

    // Tries to dealloc through a wildcard reference even though all exposed
    // references are disabled.

    unsafe { std::alloc::dealloc(wild as *mut u8, Layout::new::<u32>()) }; //~ ERROR: /deallocation through <wildcard> .* is forbidden/
}
