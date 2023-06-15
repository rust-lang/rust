// This should fail even without validation, but some MIR opts mask the error
//@compile-flags: -Zmiri-disable-validation -Zmir-opt-level=0 -Zmiri-permissive-provenance

static mut LEAK: usize = 0;

fn fill(v: &mut i32) {
    unsafe {
        LEAK = v as *mut _ as usize;
    }
}

fn evil() {
    let _ = unsafe { &mut *(LEAK as *mut i32) }; //~ ERROR: is a dangling pointer
}

fn main() {
    let _y;
    {
        let mut x = 0i32;
        fill(&mut x);
        _y = x;
    }
    // Now we use a pointer to `x` which is no longer in scope, and thus dead (even though the
    // `main` stack frame still exists). We even try going through a `usize` for extra sneakiness!
    evil();
}
