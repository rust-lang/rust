#![warn(clippy::unnecessary_cast)]
//@no-rustfix
fn main() {
    let _ = std::ptr::null() as *const u8;
    //~^ unnecessary_cast
}

mod issue11113 {
    #[repr(C)]
    struct Vtbl {
        query: unsafe extern "system" fn(),
    }

    struct TearOff {
        object: *mut std::ffi::c_void,
    }

    impl TearOff {
        unsafe fn query(&self) {
            unsafe {
                ((*(*(self.object as *mut *mut _) as *mut Vtbl)).query)()
                //~^ unnecessary_cast
            }
        }
    }
}
