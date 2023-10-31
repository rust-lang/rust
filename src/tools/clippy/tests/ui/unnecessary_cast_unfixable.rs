#![warn(clippy::unnecessary_cast)]

fn main() {
    let _ = std::ptr::null() as *const u8;
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
            ((*(*(self.object as *mut *mut _) as *mut Vtbl)).query)()
        }
    }
}
