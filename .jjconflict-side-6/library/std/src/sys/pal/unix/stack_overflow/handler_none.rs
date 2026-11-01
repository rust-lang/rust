pub unsafe fn init() {}

pub fn make_handler(_main_thread: bool) -> super::Handler {
    super::Handler::null()
}

pub unsafe fn drop_handler(_data: *mut libc::c_void) {}
