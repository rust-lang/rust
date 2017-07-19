use std::mem::transmute;

#[allow(mutable_transmutes)]
fn main() {
    unsafe {
        let bs = b"this is a test";
        transmute::<&[u8], &mut [u8]>(bs)[4] = 42; //~ ERROR: tried to modify constant memory
    }
}
