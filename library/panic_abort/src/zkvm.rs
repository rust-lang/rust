use alloc::string::String;
use core::panic::PanicPayload;

/// Note this function works with both `zkvm` and `succinct-zkvm`.
pub(crate) unsafe fn zkvm_set_abort_message(payload: &mut dyn PanicPayload) {
    let payload = payload.get();
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

    unsafe extern "C" {
        fn sys_panic(msg_ptr: *const u8, len: usize) -> !;
    }

    unsafe {
        sys_panic(msg.as_ptr(), msg.len());
    }
}
