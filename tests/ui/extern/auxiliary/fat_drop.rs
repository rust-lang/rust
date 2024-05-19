pub static mut DROPPED: bool = false;

pub struct S {
    _unsized: [u8]
}

impl Drop for S {
    fn drop(&mut self) {
        unsafe {
            DROPPED = true;
        }
    }
}
