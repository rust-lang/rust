use crate::mem;

#[thread_local]
static mut DTORS: Vec<(*mut u8, unsafe extern "C" fn(*mut u8))> = Vec::new();

pub unsafe fn register_dtor(t: *mut u8, dtor: unsafe extern "C" fn(*mut u8)) {
    let dtors = unsafe { &mut DTORS };
    dtors.push((t, dtor));
}

pub(super) unsafe fn run_dtors() {
    let mut dtors = mem::take(unsafe { &mut DTORS });
    while !dtors.is_empty() {
        for (t, dtor) in dtors {
            unsafe {
                dtor(t);
            }
        }

        dtors = mem::take(unsafe { &mut DTORS });
    }
}
