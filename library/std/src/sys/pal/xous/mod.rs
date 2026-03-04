#![forbid(unsafe_op_in_unsafe_fn)]

pub mod os;
pub mod params;

#[path = "../unsupported/common.rs"]
mod common;
pub use common::*;

#[cfg(not(test))]
#[cfg(feature = "panic-unwind")]
mod eh_unwinding {
    pub(crate) struct EhFrameFinder;
    pub(crate) static mut EH_FRAME_ADDRESS: usize = 0;
    pub(crate) static EH_FRAME_SETTINGS: EhFrameFinder = EhFrameFinder;

    unsafe impl unwind::EhFrameFinder for EhFrameFinder {
        fn find(&self, _pc: usize) -> Option<unwind::FrameInfo> {
            if unsafe { EH_FRAME_ADDRESS == 0 } {
                None
            } else {
                Some(unwind::FrameInfo {
                    text_base: None,
                    kind: unwind::FrameInfoKind::EhFrame(unsafe { EH_FRAME_ADDRESS }),
                })
            }
        }
    }
}

#[cfg(not(test))]
mod c_compat {
    use crate::os::xous::ffi::exit;

    unsafe extern "C" {
        fn main() -> u32;
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn abort() {
        exit(1);
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn _start(eh_frame: usize, params: *mut u8) {
        #[cfg(feature = "panic-unwind")]
        {
            unsafe { super::eh_unwinding::EH_FRAME_ADDRESS = eh_frame };
            unwind::set_custom_eh_frame_finder(&super::eh_unwinding::EH_FRAME_SETTINGS).ok();
        }

        unsafe { super::params::set(params) };
        exit(unsafe { main() });
    }
}
