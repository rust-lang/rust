use crate::ops::Range;

pub fn init(_guard_page_range: Option<Range<usize>>) {}

pub fn make_handler(_main_thread: bool) -> super::Handler {
    super::Handler::null()
}

pub unsafe fn drop_handler(_data: *mut libc::c_void) {}
