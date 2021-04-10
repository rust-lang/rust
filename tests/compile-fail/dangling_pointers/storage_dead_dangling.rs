// This should fail even without validation, but some MIR opts mask the error
// compile-flags: -Zmiri-disable-validation -Zmir-opt-level=0

static mut LEAK: usize = 0;

fn fill(v: &mut i32) {
    unsafe { LEAK = v as *mut _ as usize; }
}

fn evil() {
    unsafe { &mut *(LEAK as *mut i32) }; //~ ERROR dereferenced after this allocation got freed
}

fn main() {
    let _y;
    {
        let mut x = 0i32;
        fill(&mut x);
        _y = x;
    }
    // Now we use a pointer to `x` which is no longer in scope, and thus dead (even though the
    // `main` stack frame still exists).
    evil();
}
