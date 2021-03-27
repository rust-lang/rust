use alloc::string::String;
use alloc::vec::Vec;
use core::mem::transmute;
use core::panic::BoxMeUp;

const ANDROID_SET_ABORT_MESSAGE: &[u8] = b"android_set_abort_message\0";
type SetAbortMessageType = unsafe extern "C" fn(*const libc::c_char) -> ();

// Forward the abort message to libc's android_set_abort_message. The fallible allocator is used
// to avoid panicking, as this function may already be called as part of a failed allocation.
//
// Weakly resolve the symbol for android_set_abort_message. This function is only available
// for API >= 21.
pub(crate) unsafe fn android_set_abort_message(payload: *mut &mut dyn BoxMeUp) {
    let func_addr =
        libc::dlsym(libc::RTLD_DEFAULT, ANDROID_SET_ABORT_MESSAGE.as_ptr() as *const libc::c_char)
            as usize;
    if func_addr == 0 {
        return;
    }

    let payload = (*payload).get();
    let msg = match payload.downcast_ref::<&'static str>() {
        Some(msg) => msg.as_bytes(),
        None => match payload.downcast_ref::<String>() {
            Some(msg) => msg.as_bytes(),
            None => &[],
        },
    };
    if msg.is_empty() {
        return;
    }

    let size = msg.len() + 1;
    let mut v = Vec::new();
    if v.try_reserve(size).is_err() {
        return;
    }

    v.extend(msg);
    v.push(0);
    let func = transmute::<usize, SetAbortMessageType>(func_addr);
    func(v.as_ptr() as *const libc::c_char);
}
