fn main() {
    let mut data: [u8; 1024] = unsafe { std::mem::uninitialized() };
    for i in 0..data.len() {
        unsafe { std::ptr::write(&mut data[i], 0); }
    }
    assert_eq!(data.len(), 1024);
}
