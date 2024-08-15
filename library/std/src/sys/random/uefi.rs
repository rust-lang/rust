use r_efi::protocols::rng;

use crate::sys::pal::helpers;

pub fn fill_bytes(bytes: &mut [u8]) {
    let handles =
        helpers::locate_handles(rng::PROTOCOL_GUID).expect("failed to generate random data");
    for handle in handles {
        if let Ok(protocol) = helpers::open_protocol::<rng::Protocol>(handle, rng::PROTOCOL_GUID) {
            let r = unsafe {
                ((*protocol.as_ptr()).get_rng)(
                    protocol.as_ptr(),
                    crate::ptr::null_mut(),
                    bytes.len(),
                    bytes.as_mut_ptr(),
                )
            };
            if r.is_error() {
                continue;
            } else {
                return;
            }
        }
    }

    panic!("failed to generate random data");
}
