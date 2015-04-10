/// darwin_fd_limit exists to work around an issue where launchctl on Mac OS X
/// defaults the rlimit maxfiles to 256/unlimited. The default soft limit of 256
/// ends up being far too low for our multithreaded scheduler testing, depending
/// on the number of cores available.
///
/// This fixes issue #7772.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(non_camel_case_types)]
pub fn raise_fd_limit() {
    use libc;

    type rlim_t = libc::uint64_t;
    #[repr(C)]
    struct rlimit {
        rlim_cur: rlim_t,
        rlim_max: rlim_t
    }
    extern {
        // name probably doesn't need to be mut, but the C function doesn't
        // specify const
        fn sysctl(name: *mut libc::c_int, namelen: libc::c_uint,
                  oldp: *mut libc::c_void, oldlenp: *mut libc::size_t,
                  newp: *mut libc::c_void, newlen: libc::size_t) -> libc::c_int;
        fn getrlimit(resource: libc::c_int, rlp: *mut rlimit) -> libc::c_int;
        fn setrlimit(resource: libc::c_int, rlp: *const rlimit) -> libc::c_int;
    }
    static CTL_KERN: libc::c_int = 1;
    static KERN_MAXFILESPERPROC: libc::c_int = 29;
    static RLIMIT_NOFILE: libc::c_int = 8;

    // The strategy here is to fetch the current resource limits, read the
    // kern.maxfilesperproc sysctl value, and bump the soft resource limit for
    // maxfiles up to the sysctl value.
    use ptr::null_mut;
    use mem::size_of_val;
    use io;

    // Fetch the kern.maxfilesperproc value
    let mut mib: [libc::c_int; 2] = [CTL_KERN, KERN_MAXFILESPERPROC];
    let mut maxfiles: libc::c_int = 0;
    let mut size: libc::size_t = size_of_val(&maxfiles) as libc::size_t;
    if sysctl(&mut mib[0], 2, &mut maxfiles as *mut _ as *mut _, &mut size,
              null_mut(), 0) != 0 {
        let err = io::Error::last_os_error();
        panic!("raise_fd_limit: error calling sysctl: {}", err);
    }

    // Fetch the current resource limits
    let mut rlim = rlimit{rlim_cur: 0, rlim_max: 0};
    if getrlimit(RLIMIT_NOFILE, &mut rlim) != 0 {
        let err = io::Error::last_os_error();
        panic!("raise_fd_limit: error calling getrlimit: {}", err);
    }

    // Bump the soft limit to the smaller of kern.maxfilesperproc and the hard
    // limit
    rlim.rlim_cur = ::cmp::min(maxfiles as rlim_t, rlim.rlim_max);

    // Set our newly-increased resource limit
    if setrlimit(RLIMIT_NOFILE, &rlim) != 0 {
        let err = io::Error::last_os_error();
        panic!("raise_fd_limit: error calling setrlimit: {}", err);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn raise_fd_limit() {}
