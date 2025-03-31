#![allow(unnecessary_transmutes)]
pub fn main() {
    let bytes: [u8; 8] = unsafe { ::std::mem::transmute(0u64) };
    let _val: &[u8] = &bytes;
}
