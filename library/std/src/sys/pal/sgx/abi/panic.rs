use super::usercalls::alloc::UserRef;
use crate::io::{self, Write};
use crate::{cmp, mem};

extern "C" {
    fn take_debug_panic_buf_ptr() -> *mut u8;
    static DEBUG: u8;
}

pub(crate) struct SgxPanicOutput(Option<&'static mut UserRef<[u8]>>);

fn empty_user_slice() -> &'static mut UserRef<[u8]> {
    unsafe { UserRef::from_raw_parts_mut(1 as *mut u8, 0) }
}

impl SgxPanicOutput {
    pub(crate) fn new() -> Option<Self> {
        if unsafe { DEBUG == 0 } { None } else { Some(SgxPanicOutput(None)) }
    }

    fn init(&mut self) -> &mut &'static mut UserRef<[u8]> {
        self.0.get_or_insert_with(|| unsafe {
            let ptr = take_debug_panic_buf_ptr();
            if ptr.is_null() { empty_user_slice() } else { UserRef::from_raw_parts_mut(ptr, 1024) }
        })
    }
}

impl Write for SgxPanicOutput {
    fn write(&mut self, src: &[u8]) -> io::Result<usize> {
        let dst = mem::replace(self.init(), empty_user_slice());
        let written = cmp::min(src.len(), dst.len());
        dst[..written].copy_from_enclave(&src[..written]);
        self.0 = Some(&mut dst[written..]);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
