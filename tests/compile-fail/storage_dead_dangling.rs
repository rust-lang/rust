// This should fail even without validation
// compile-flags: -Zmiri-disable-validation

static mut LEAK: usize = 0;

fn fill(v: &mut i32) {
    unsafe { LEAK = v as *mut _ as usize; }
}

fn evil() {
    unsafe { &mut *(LEAK as *mut i32) }; //~ ERROR dangling pointer was dereferenced
}

fn main() {
    let _y;
    {
        let mut x = 0i32;
        fill(&mut x);
        _y = x;
    }
    evil();
}
