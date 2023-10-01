#![warn(clippy::unnecessary_cast)]
//@no-rustfix
fn main() {
    let _ = std::ptr::null() as *const u8;
    //~^ ERROR: casting raw pointers to the same type and constness is unnecessary (`*cons
    //~| NOTE: `-D clippy::unnecessary-cast` implied by `-D warnings`
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
            //~^ ERROR: casting raw pointers to the same type and constness is unnecessary
        }
    }
}
