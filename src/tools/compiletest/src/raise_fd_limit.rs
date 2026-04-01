/// darwin_fd_limit exists to work around an issue where launchctl on macOS
/// defaults the rlimit maxfiles to 256/unlimited. The default soft limit of 256
/// ends up being far too low for our multithreaded scheduler testing, depending
/// on the number of cores available.
///
/// This fixes issue #7772.
#[cfg(target_vendor = "apple")]
#[allow(non_camel_case_types)]
// FIXME(#139616): document caller contract.
pub unsafe fn raise_fd_limit() {
    use std::ptr::null_mut;
    use std::{cmp, io};

    static CTL_KERN: libc::c_int = 1;
    static KERN_MAXFILESPERPROC: libc::c_int = 29;

    // The strategy here is to fetch the current resource limits, read the
    // kern.maxfilesperproc sysctl value, and bump the soft resource limit for
    // maxfiles up to the sysctl value.

    // Fetch the kern.maxfilesperproc value
    let mut mib: [libc::c_int; 2] = [CTL_KERN, KERN_MAXFILESPERPROC];
    let mut maxfiles: libc::c_int = 0;
    let mut size: libc::size_t = size_of_val(&maxfiles) as libc::size_t;
    // FIXME(#139616): justify why this is sound.
    if unsafe {
        libc::sysctl(&mut mib[0], 2, &mut maxfiles as *mut _ as *mut _, &mut size, null_mut(), 0)
    } != 0
    {
        let err = io::Error::last_os_error();
        panic!("raise_fd_limit: error calling sysctl: {}", err);
    }

    // Fetch the current resource limits
    let mut rlim = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    // FIXME(#139616): justify why this is sound.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut rlim) } != 0 {
        let err = io::Error::last_os_error();
        panic!("raise_fd_limit: error calling getrlimit: {}", err);
    }

    // Make sure we're only ever going to increase the rlimit.
    if rlim.rlim_cur < maxfiles as libc::rlim_t {
        // Bump the soft limit to the smaller of kern.maxfilesperproc and the hard limit.
        rlim.rlim_cur = cmp::min(maxfiles as libc::rlim_t, rlim.rlim_max);

        // Set our newly-increased resource limit.
        // FIXME(#139616): justify why this is sound.
        if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &rlim) } != 0 {
            let err = io::Error::last_os_error();
            panic!("raise_fd_limit: error calling setrlimit: {}", err);
        }
    }
}

#[cfg(not(target_vendor = "apple"))]
pub unsafe fn raise_fd_limit() {}
