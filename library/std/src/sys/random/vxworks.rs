use crate::sync::atomic::AtomicBool;
use crate::sync::atomic::Ordering::Relaxed;

static RNG_INIT: AtomicBool = AtomicBool::new(false);

pub fn fill_bytes(mut bytes: &mut [u8]) {
    while !RNG_INIT.load(Relaxed) {
        let ret = unsafe { libc::randSecure() };
        if ret < 0 {
            panic!("failed to generate random data");
        } else if ret > 0 {
            RNG_INIT.store(true, Relaxed);
            break;
        }

        unsafe { libc::usleep(10) };
    }

    while !bytes.is_empty() {
        let len = bytes.len().try_into().unwrap_or(libc::c_int::MAX);
        let ret = unsafe { libc::randABytes(bytes.as_mut_ptr(), len) };
        assert!(ret >= 0, "failed to generate random data");
        bytes = &mut bytes[len as usize..];
    }
}
