use crate::ops::Range;

mod c {
    pub type PVECTORED_EXCEPTION_HANDLER =
        Option<unsafe extern "system" fn(exceptioninfo: *mut EXCEPTION_POINTERS) -> i32>;
    pub type NTSTATUS = i32;
    pub type BOOL = i32;

    unsafe extern "system" {
        pub fn AddVectoredExceptionHandler(
            first: u32,
            handler: PVECTORED_EXCEPTION_HANDLER,
        ) -> *mut core::ffi::c_void;
        pub fn SetThreadStackGuarantee(stacksizeinbytes: *mut u32) -> BOOL;
    }

    pub const EXCEPTION_STACK_OVERFLOW: NTSTATUS = 0xC00000FD_u32 as _;
    pub const EXCEPTION_CONTINUE_SEARCH: i32 = 1i32;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct EXCEPTION_POINTERS {
        pub ExceptionRecord: *mut EXCEPTION_RECORD,
        // We don't need this field here
        // pub Context: *mut CONTEXT,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct EXCEPTION_RECORD {
        pub ExceptionCode: NTSTATUS,
        pub ExceptionFlags: u32,
        pub ExceptionRecord: *mut EXCEPTION_RECORD,
        pub ExceptionAddress: *mut core::ffi::c_void,
        pub NumberParameters: u32,
        pub ExceptionInformation: [usize; 15],
    }
}

/// Reserve stack space for use in stack overflow exceptions.
fn reserve_stack() {
    let result = unsafe { c::SetThreadStackGuarantee(&mut 0x5000) };
    // Reserving stack space is not critical so we allow it to fail in the released build of libstd.
    // We still use debug assert here so that CI will test that we haven't made a mistake calling the function.
    debug_assert_ne!(result, 0, "failed to reserve stack space for exception handling");
}

unsafe extern "system" fn vectored_handler(ExceptionInfo: *mut c::EXCEPTION_POINTERS) -> i32 {
    // SAFETY: It's up to the caller (which in this case is the OS) to ensure that `ExceptionInfo` is valid.
    unsafe {
        let rec = &(*(*ExceptionInfo).ExceptionRecord);
        let code = rec.ExceptionCode;

        if code == c::EXCEPTION_STACK_OVERFLOW {
            crate::thread::with_current_name(|name| {
                let name = name.unwrap_or("<unknown>");
                let tid = crate::thread::current_os_id();
                rtprintpanic!("\nthread '{name}' ({tid}) has overflowed its stack\n");
            });
        }
        c::EXCEPTION_CONTINUE_SEARCH
    }
}

pub fn init(_guard_page_range: Option<Range<usize>>) {
    // SAFETY: `vectored_handler` has the correct ABI and is safe to call during exception handling.
    unsafe {
        let result = c::AddVectoredExceptionHandler(0, Some(vectored_handler));
        // Similar to the above, adding the stack overflow handler is allowed to fail
        // but a debug assert is used so CI will still test that it normally works.
        debug_assert!(!result.is_null(), "failed to install exception handler");
    }
    // Set the thread stack guarantee for the main thread.
    reserve_stack();
}

pub fn make_handler(main_thread: bool) -> super::Handler {
    if !main_thread {
        reserve_stack();
    }
    super::Handler::null()
}

pub unsafe fn drop_handler(_data: *mut libc::c_void) {}
