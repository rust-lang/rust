use std::mem;

fn main() {
    #[allow(deprecated)]
    let xs: [u8; 4] = unsafe { mem::uninitialized() };
    let y = xs[0] + xs[1];
}
