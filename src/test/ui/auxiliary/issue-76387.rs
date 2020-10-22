// compile-flags: -C opt-level=3

pub struct FatPtr {
    ptr: *mut u8,
    len: usize,
}

impl FatPtr {
    pub fn new(len: usize) -> FatPtr {
        let ptr = Box::into_raw(vec![42u8; len].into_boxed_slice()) as *mut u8;

        FatPtr { ptr, len }
    }
}

impl std::ops::Deref for FatPtr {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::ops::Drop for FatPtr {
    fn drop(&mut self) {
        println!("Drop");
    }
}
