use std::mem;

fn dangling() -> *const u8 {
    let x = 0u8;
    &x as *const _
}

fn main() {
    let _x: &i32 = unsafe { mem::transmute(dangling()) }; //~ ERROR dangling reference (use-after-free)
}
