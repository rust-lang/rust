use super::usercalls::alloc::UserMut;
use crate::io::{self, Write};
use crate::{cmp, mem};

unsafe extern "C" {
    fn take_debug_panic_buf_ptr() -> *mut u8;
    static DEBUG: u8;
}

pub(crate) struct SgxPanicOutput(Option<UserMut<'static, [u8]>>);

fn empty_user_slice() -> UserMut<'static, [u8]> {
    unsafe { UserMut::from_raw_parts_mut(1 as *mut u8, 0) }
}

impl SgxPanicOutput {
    pub(crate) fn new() -> Option<Self> {
        if unsafe { DEBUG == 0 } { None } else { Some(SgxPanicOutput(None)) }
    }

    fn init(&mut self) -> &mut UserMut<'static, [u8]> {
        self.0.get_or_insert_with(|| unsafe {
            let ptr = take_debug_panic_buf_ptr();
            if ptr.is_null() { empty_user_slice() } else { UserMut::from_raw_parts_mut(ptr, 1024) }
        })
    }
}

impl Write for SgxPanicOutput {
    fn write(&mut self, src: &[u8]) -> io::Result<usize> {
        let mut dst = mem::replace(self.init(), empty_user_slice());
        let written = cmp::min(src.len(), dst.coerce_shared().len());
        dst.reborrow().index_mut(..written).copy_from_enclave(&src[..written]);
        self.0 = Some(dst.index_mut(written..));
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
