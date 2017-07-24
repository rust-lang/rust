use std::mem::transmute;

#[allow(mutable_transmutes)]
fn main() {
    unsafe {
        let s = "this is a test";
        transmute::<&[u8], &mut [u8]>(s.as_bytes())[4] = 42; //~ ERROR: tried to modify constant memory
    }
}
