use crate::io;
use crate::num::NonZero;
use crate::ptr::NonNull;
use crate::time::Duration;

pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    // UEFI is single threaded
    Ok(NonZero::new(1).unwrap())
}

pub fn sleep(dur: Duration) {
    let boot_services: NonNull<r_efi::efi::BootServices> =
        crate::os::uefi::env::boot_services().expect("can't sleep").cast();
    let mut dur_ms = dur.as_micros();
    // ceil up to the nearest microsecond
    if dur.subsec_nanos() % 1000 > 0 {
        dur_ms += 1;
    }

    while dur_ms > 0 {
        let ms = crate::cmp::min(dur_ms, usize::MAX as u128);
        let _ = unsafe { ((*boot_services.as_ptr()).stall)(ms as usize) };
        dur_ms -= ms;
    }
}
