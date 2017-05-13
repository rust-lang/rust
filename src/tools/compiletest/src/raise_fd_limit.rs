// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// darwin_fd_limit exists to work around an issue where launchctl on Mac OS X
/// defaults the rlimit maxfiles to 256/unlimited. The default soft limit of 256
/// ends up being far too low for our multithreaded scheduler testing, depending
/// on the number of cores available.
///
/// This fixes issue #7772.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[allow(non_camel_case_types)]
pub unsafe fn raise_fd_limit() {
    use libc;
    use std::cmp;
    use std::io;
    use std::mem::size_of_val;
    use std::ptr::null_mut;

    static CTL_KERN: libc::c_int = 1;
    static KERN_MAXFILESPERPROC: libc::c_int = 29;

    // The strategy here is to fetch the current resource limits, read the
    // kern.maxfilesperproc sysctl value, and bump the soft resource limit for
    // maxfiles up to the sysctl value.

    // Fetch the kern.maxfilesperproc value
    let mut mib: [libc::c_int; 2] = [CTL_KERN, KERN_MAXFILESPERPROC];
    let mut maxfiles: libc::c_int = 0;
    let mut size: libc::size_t = size_of_val(&maxfiles) as libc::size_t;
    if libc::sysctl(&mut mib[0],
                    2,
                    &mut maxfiles as *mut _ as *mut _,
                    &mut size,
                    null_mut(),
                    0) != 0 {
        let err = io::Error::last_os_error();
        panic!("raise_fd_limit: error calling sysctl: {}", err);
    }

    // Fetch the current resource limits
    let mut rlim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if libc::getrlimit(libc::RLIMIT_NOFILE, &mut rlim) != 0 {
        let err = io::Error::last_os_error();
        panic!("raise_fd_limit: error calling getrlimit: {}", err);
    }

    // Bump the soft limit to the smaller of kern.maxfilesperproc and the hard
    // limit
    rlim.rlim_cur = cmp::min(maxfiles as libc::rlim_t, rlim.rlim_max);

    // Set our newly-increased resource limit
    if libc::setrlimit(libc::RLIMIT_NOFILE, &rlim) != 0 {
        let err = io::Error::last_os_error();
        panic!("raise_fd_limit: error calling setrlimit: {}", err);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub unsafe fn raise_fd_limit() {}
