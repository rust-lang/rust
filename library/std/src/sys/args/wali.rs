pub use super::common::Args;

/// One-time global initialization.
pub unsafe fn init(argc: isize, argv: *const *const u8) {
    unsafe { imp::init(argc, argv) }
}

/// Returns the command line arguments
pub fn args() -> Args {
    imp::args()
}

mod imp {
    use super::Args;
    use crate::ffi::{CString, OsString};
    use crate::os::raw::{c_uint as WaliArgIdx, c_uint};
    use crate::os::unix::prelude::*;
    use crate::sync::OnceLock;

    #[link(wasm_import_module = "wali")]
    unsafe extern "C" {
        pub fn __cl_get_argc() -> WaliArgIdx;
        pub fn __cl_get_argv_len(offset: WaliArgIdx) -> c_uint;
        pub fn __cl_copy_argv(buf: *mut i8, offset: WaliArgIdx) -> c_uint;
    }

    static ARGS: OnceLock<Vec<OsString>> = OnceLock::new();

    pub unsafe fn init(_argc: isize, _argv: *const *const u8) {
        // Uses the WALI arguments API
        ARGS.set(argc_argv()).ok();
    }

    unsafe fn load_arg(idx: c_uint) -> OsString {
        let arg_len = unsafe { __cl_get_argv_len(idx) };
        let arg_buf = CString::new(vec![b'x'; arg_len as usize]).unwrap();
        let ptr = arg_buf.into_raw();
        let arg_buf = unsafe {
            __cl_copy_argv(ptr, idx);
            CString::from_raw(ptr)
        };
        OsStringExt::from_vec(arg_buf.into_bytes())
    }

    fn argc_argv() -> Vec<OsString> {
        let argc = unsafe { __cl_get_argc() };
        (0..argc).map(|x| unsafe { load_arg(x) }).collect()
    }

    pub fn args() -> Args {
        let cached = ARGS.get().cloned().unwrap_or_default();
        Args::new(cached)
    }
}
