use crate::ffi::CStr;
use crate::mem::{self, ManuallyDrop};
use crate::num::NonZero;
#[cfg(all(target_os = "linux", target_env = "gnu"))]
use crate::sys::weak::dlsym;
#[cfg(any(target_os = "solaris", target_os = "illumos", target_os = "nto",))]
use crate::sys::weak::weak;
use crate::sys::{os, stack_overflow};
use crate::time::Duration;
use crate::{cmp, io, ptr};
#[cfg(not(any(target_os = "l4re", target_os = "vxworks", target_os = "espidf")))]
pub const DEFAULT_MIN_STACK_SIZE: usize = 2 * 1024 * 1024;
#[cfg(target_os = "l4re")]
pub const DEFAULT_MIN_STACK_SIZE: usize = 1024 * 1024;
#[cfg(target_os = "vxworks")]
pub const DEFAULT_MIN_STACK_SIZE: usize = 256 * 1024;
#[cfg(target_os = "espidf")]
pub const DEFAULT_MIN_STACK_SIZE: usize = 0; // 0 indicates that the stack size configured in the ESP-IDF menuconfig system should be used

#[cfg(target_os = "fuchsia")]
mod zircon {
    type zx_handle_t = u32;
    type zx_status_t = i32;
    pub const ZX_PROP_NAME: u32 = 3;

    unsafe extern "C" {
        pub fn zx_object_set_property(
            handle: zx_handle_t,
            property: u32,
            value: *const libc::c_void,
            value_size: libc::size_t,
        ) -> zx_status_t;
        pub fn zx_thread_self() -> zx_handle_t;
    }
}

pub struct Thread {
    id: libc::pthread_t,
}

// Some platforms may have pthread_t as a pointer in which case we still want
// a thread to be Send/Sync
unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        let p = Box::into_raw(Box::new(p));
        let mut native: libc::pthread_t = mem::zeroed();
        let mut attr: mem::MaybeUninit<libc::pthread_attr_t> = mem::MaybeUninit::uninit();
        assert_eq!(libc::pthread_attr_init(attr.as_mut_ptr()), 0);

        #[cfg(target_os = "espidf")]
        if stack > 0 {
            // Only set the stack if a non-zero value is passed
            // 0 is used as an indication that the default stack size configured in the ESP-IDF menuconfig system should be used
            assert_eq!(
                libc::pthread_attr_setstacksize(
                    attr.as_mut_ptr(),
                    cmp::max(stack, min_stack_size(&attr))
                ),
                0
            );
        }

        #[cfg(not(target_os = "espidf"))]
        {
            let stack_size = cmp::max(stack, min_stack_size(attr.as_ptr()));

            match libc::pthread_attr_setstacksize(attr.as_mut_ptr(), stack_size) {
                0 => {}
                n => {
                    assert_eq!(n, libc::EINVAL);
                    // EINVAL means |stack_size| is either too small or not a
                    // multiple of the system page size. Because it's definitely
                    // >= PTHREAD_STACK_MIN, it must be an alignment issue.
                    // Round up to the nearest page and try again.
                    let page_size = os::page_size();
                    let stack_size =
                        (stack_size + page_size - 1) & (-(page_size as isize - 1) as usize - 1);
                    assert_eq!(libc::pthread_attr_setstacksize(attr.as_mut_ptr(), stack_size), 0);
                }
            };
        }

        let ret = libc::pthread_create(&mut native, attr.as_ptr(), thread_start, p as *mut _);
        // Note: if the thread creation fails and this assert fails, then p will
        // be leaked. However, an alternative design could cause double-free
        // which is clearly worse.
        assert_eq!(libc::pthread_attr_destroy(attr.as_mut_ptr()), 0);

        return if ret != 0 {
            // The thread failed to start and as a result p was not consumed. Therefore, it is
            // safe to reconstruct the box so that it gets deallocated.
            drop(Box::from_raw(p));
            Err(io::Error::from_raw_os_error(ret))
        } else {
            Ok(Thread { id: native })
        };

        extern "C" fn thread_start(main: *mut libc::c_void) -> *mut libc::c_void {
            unsafe {
                // Next, set up our stack overflow handler which may get triggered if we run
                // out of stack.
                let _handler = stack_overflow::Handler::new();
                // Finally, let's run some code.
                Box::from_raw(main as *mut Box<dyn FnOnce()>)();
            }
            ptr::null_mut()
        }
    }

    pub fn yield_now() {
        let ret = unsafe { libc::sched_yield() };
        debug_assert_eq!(ret, 0);
    }

    #[cfg(target_os = "android")]
    pub fn set_name(name: &CStr) {
        const PR_SET_NAME: libc::c_int = 15;
        unsafe {
            let res = libc::prctl(
                PR_SET_NAME,
                name.as_ptr(),
                0 as libc::c_ulong,
                0 as libc::c_ulong,
                0 as libc::c_ulong,
            );
            // We have no good way of propagating errors here, but in debug-builds let's check that this actually worked.
            debug_assert_eq!(res, 0);
        }
    }

    #[cfg(any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "nuttx"
    ))]
    pub fn set_name(name: &CStr) {
        unsafe {
            cfg_if::cfg_if! {
                if #[cfg(target_os = "linux")] {
                    // Linux limits the allowed length of the name.
                    const TASK_COMM_LEN: usize = 16;
                    let name = truncate_cstr::<{ TASK_COMM_LEN }>(name);
                } else {
                    // FreeBSD, DragonFly, FreeBSD and NuttX do not enforce length limits.
                }
            };
            // Available since glibc 2.12, musl 1.1.16, and uClibc 1.0.20 for Linux,
            // FreeBSD 12.2 and 13.0, and DragonFly BSD 6.0.
            let res = libc::pthread_setname_np(libc::pthread_self(), name.as_ptr());
            // We have no good way of propagating errors here, but in debug-builds let's check that this actually worked.
            debug_assert_eq!(res, 0);
        }
    }

    #[cfg(target_os = "openbsd")]
    pub fn set_name(name: &CStr) {
        unsafe {
            libc::pthread_set_name_np(libc::pthread_self(), name.as_ptr());
        }
    }

    #[cfg(target_vendor = "apple")]
    pub fn set_name(name: &CStr) {
        unsafe {
            let name = truncate_cstr::<{ libc::MAXTHREADNAMESIZE }>(name);
            let res = libc::pthread_setname_np(name.as_ptr());
            // We have no good way of propagating errors here, but in debug-builds let's check that this actually worked.
            debug_assert_eq!(res, 0);
        }
    }

    #[cfg(target_os = "netbsd")]
    pub fn set_name(name: &CStr) {
        unsafe {
            let res = libc::pthread_setname_np(
                libc::pthread_self(),
                c"%s".as_ptr(),
                name.as_ptr() as *mut libc::c_void,
            );
            debug_assert_eq!(res, 0);
        }
    }

    #[cfg(any(target_os = "solaris", target_os = "illumos", target_os = "nto"))]
    pub fn set_name(name: &CStr) {
        weak! {
            fn pthread_setname_np(
                libc::pthread_t, *const libc::c_char
            ) -> libc::c_int
        }

        if let Some(f) = pthread_setname_np.get() {
            #[cfg(target_os = "nto")]
            const THREAD_NAME_MAX: usize = libc::_NTO_THREAD_NAME_MAX as usize;
            #[cfg(any(target_os = "solaris", target_os = "illumos"))]
            const THREAD_NAME_MAX: usize = 32;

            let name = truncate_cstr::<{ THREAD_NAME_MAX }>(name);
            let res = unsafe { f(libc::pthread_self(), name.as_ptr()) };
            debug_assert_eq!(res, 0);
        }
    }

    #[cfg(target_os = "fuchsia")]
    pub fn set_name(name: &CStr) {
        use self::zircon::*;
        unsafe {
            zx_object_set_property(
                zx_thread_self(),
                ZX_PROP_NAME,
                name.as_ptr() as *const libc::c_void,
                name.to_bytes().len(),
            );
        }
    }

    #[cfg(target_os = "haiku")]
    pub fn set_name(name: &CStr) {
        unsafe {
            let thread_self = libc::find_thread(ptr::null_mut());
            let res = libc::rename_thread(thread_self, name.as_ptr());
            // We have no good way of propagating errors here, but in debug-builds let's check that this actually worked.
            debug_assert_eq!(res, libc::B_OK);
        }
    }

    #[cfg(target_os = "vxworks")]
    pub fn set_name(name: &CStr) {
        // FIXME(libc): adding real STATUS, ERROR type eventually.
        unsafe extern "C" {
            fn taskNameSet(task_id: libc::TASK_ID, task_name: *mut libc::c_char) -> libc::c_int;
        }

        //  VX_TASK_NAME_LEN is 31 in VxWorks 7.
        const VX_TASK_NAME_LEN: usize = 31;

        let mut name = truncate_cstr::<{ VX_TASK_NAME_LEN }>(name);
        let res = unsafe { taskNameSet(libc::taskIdSelf(), name.as_mut_ptr()) };
        debug_assert_eq!(res, libc::OK);
    }

    #[cfg(any(
        target_env = "newlib",
        target_os = "l4re",
        target_os = "emscripten",
        target_os = "redox",
        target_os = "hurd",
        target_os = "aix",
    ))]
    pub fn set_name(_name: &CStr) {
        // Newlib and Emscripten have no way to set a thread name.
    }

    #[cfg(not(target_os = "espidf"))]
    pub fn sleep(dur: Duration) {
        let mut secs = dur.as_secs();
        let mut nsecs = dur.subsec_nanos() as _;

        // If we're awoken with a signal then the return value will be -1 and
        // nanosleep will fill in `ts` with the remaining time.
        unsafe {
            while secs > 0 || nsecs > 0 {
                let mut ts = libc::timespec {
                    tv_sec: cmp::min(libc::time_t::MAX as u64, secs) as libc::time_t,
                    tv_nsec: nsecs,
                };
                secs -= ts.tv_sec as u64;
                let ts_ptr = &raw mut ts;
                if libc::nanosleep(ts_ptr, ts_ptr) == -1 {
                    assert_eq!(os::errno(), libc::EINTR);
                    secs += ts.tv_sec as u64;
                    nsecs = ts.tv_nsec;
                } else {
                    nsecs = 0;
                }
            }
        }
    }

    #[cfg(target_os = "espidf")]
    pub fn sleep(dur: Duration) {
        // ESP-IDF does not have `nanosleep`, so we use `usleep` instead.
        // As per the documentation of `usleep`, it is expected to support
        // sleep times as big as at least up to 1 second.
        //
        // ESP-IDF does support almost up to `u32::MAX`, but due to a potential integer overflow in its
        // `usleep` implementation
        // (https://github.com/espressif/esp-idf/blob/d7ca8b94c852052e3bc33292287ef4dd62c9eeb1/components/newlib/time.c#L210),
        // we limit the sleep time to the maximum one that would not cause the underlying `usleep` implementation to overflow
        // (`portTICK_PERIOD_MS` can be anything between 1 to 1000, and is 10 by default).
        const MAX_MICROS: u32 = u32::MAX - 1_000_000 - 1;

        // Add any nanoseconds smaller than a microsecond as an extra microsecond
        // so as to comply with the `std::thread::sleep` contract which mandates
        // implementations to sleep for _at least_ the provided `dur`.
        // We can't overflow `micros` as it is a `u128`, while `Duration` is a pair of
        // (`u64` secs, `u32` nanos), where the nanos are strictly smaller than 1 second
        // (i.e. < 1_000_000_000)
        let mut micros = dur.as_micros() + if dur.subsec_nanos() % 1_000 > 0 { 1 } else { 0 };

        while micros > 0 {
            let st = if micros > MAX_MICROS as u128 { MAX_MICROS } else { micros as u32 };
            unsafe {
                libc::usleep(st);
            }

            micros -= st as u128;
        }
    }

    pub fn join(self) {
        let id = self.into_id();
        let ret = unsafe { libc::pthread_join(id, ptr::null_mut()) };
        assert!(ret == 0, "failed to join thread: {}", io::Error::from_raw_os_error(ret));
    }

    pub fn id(&self) -> libc::pthread_t {
        self.id
    }

    pub fn into_id(self) -> libc::pthread_t {
        ManuallyDrop::new(self).id
    }
}

impl Drop for Thread {
    fn drop(&mut self) {
        let ret = unsafe { libc::pthread_detach(self.id) };
        debug_assert_eq!(ret, 0);
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "nto",
    target_os = "solaris",
    target_os = "illumos",
    target_os = "vxworks",
    target_vendor = "apple",
))]
fn truncate_cstr<const MAX_WITH_NUL: usize>(cstr: &CStr) -> [libc::c_char; MAX_WITH_NUL] {
    let mut result = [0; MAX_WITH_NUL];
    for (src, dst) in cstr.to_bytes().iter().zip(&mut result[..MAX_WITH_NUL - 1]) {
        *dst = *src as libc::c_char;
    }
    result
}

pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    cfg_if::cfg_if! {
        if #[cfg(any(
            target_os = "android",
            target_os = "emscripten",
            target_os = "fuchsia",
            target_os = "hurd",
            target_os = "linux",
            target_os = "aix",
            target_vendor = "apple",
        ))] {
            #[allow(unused_assignments)]
            #[allow(unused_mut)]
            let mut quota = usize::MAX;

            #[cfg(any(target_os = "android", target_os = "linux"))]
            {
                quota = cgroups::quota().max(1);
                let mut set: libc::cpu_set_t = unsafe { mem::zeroed() };
                unsafe {
                    if libc::sched_getaffinity(0, mem::size_of::<libc::cpu_set_t>(), &mut set) == 0 {
                        let count = libc::CPU_COUNT(&set) as usize;
                        let count = count.min(quota);

                        // According to sched_getaffinity's API it should always be non-zero, but
                        // some old MIPS kernels were buggy and zero-initialized the mask if
                        // none was explicitly set.
                        // In that case we use the sysconf fallback.
                        if let Some(count) = NonZero::new(count) {
                            return Ok(count)
                        }
                    }
                }
            }
            match unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) } {
                -1 => Err(io::Error::last_os_error()),
                0 => Err(io::Error::UNKNOWN_THREAD_COUNT),
                cpus => {
                    let count = cpus as usize;
                    // Cover the unusual situation where we were able to get the quota but not the affinity mask
                    let count = count.min(quota);
                    Ok(unsafe { NonZero::new_unchecked(count) })
                }
            }
        } else if #[cfg(any(
                   target_os = "freebsd",
                   target_os = "dragonfly",
                   target_os = "openbsd",
                   target_os = "netbsd",
               ))] {
            use crate::ptr;

            #[cfg(target_os = "freebsd")]
            {
                let mut set: libc::cpuset_t = unsafe { mem::zeroed() };
                unsafe {
                    if libc::cpuset_getaffinity(
                        libc::CPU_LEVEL_WHICH,
                        libc::CPU_WHICH_PID,
                        -1,
                        mem::size_of::<libc::cpuset_t>(),
                        &mut set,
                    ) == 0 {
                        let count = libc::CPU_COUNT(&set) as usize;
                        if count > 0 {
                            return Ok(NonZero::new_unchecked(count));
                        }
                    }
                }
            }

            #[cfg(target_os = "netbsd")]
            {
                unsafe {
                    let set = libc::_cpuset_create();
                    if !set.is_null() {
                        let mut count: usize = 0;
                        if libc::pthread_getaffinity_np(libc::pthread_self(), libc::_cpuset_size(set), set) == 0 {
                            for i in 0..libc::cpuid_t::MAX {
                                match libc::_cpuset_isset(i, set) {
                                    -1 => break,
                                    0 => continue,
                                    _ => count = count + 1,
                                }
                            }
                        }
                        libc::_cpuset_destroy(set);
                        if let Some(count) = NonZero::new(count) {
                            return Ok(count);
                        }
                    }
                }
            }

            let mut cpus: libc::c_uint = 0;
            let mut cpus_size = crate::mem::size_of_val(&cpus);

            unsafe {
                cpus = libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as libc::c_uint;
            }

            // Fallback approach in case of errors or no hardware threads.
            if cpus < 1 {
                let mut mib = [libc::CTL_HW, libc::HW_NCPU, 0, 0];
                let res = unsafe {
                    libc::sysctl(
                        mib.as_mut_ptr(),
                        2,
                        (&raw mut cpus) as *mut _,
                        (&raw mut cpus_size) as *mut _,
                        ptr::null_mut(),
                        0,
                    )
                };

                // Handle errors if any.
                if res == -1 {
                    return Err(io::Error::last_os_error());
                } else if cpus == 0 {
                    return Err(io::Error::UNKNOWN_THREAD_COUNT);
                }
            }

            Ok(unsafe { NonZero::new_unchecked(cpus as usize) })
        } else if #[cfg(target_os = "nto")] {
            unsafe {
                use libc::_syspage_ptr;
                if _syspage_ptr.is_null() {
                    Err(io::const_error!(io::ErrorKind::NotFound, "No syspage available"))
                } else {
                    let cpus = (*_syspage_ptr).num_cpu;
                    NonZero::new(cpus as usize)
                        .ok_or(io::Error::UNKNOWN_THREAD_COUNT)
                }
            }
        } else if #[cfg(any(target_os = "solaris", target_os = "illumos"))] {
            let mut cpus = 0u32;
            if unsafe { libc::pset_info(libc::PS_MYID, core::ptr::null_mut(), &mut cpus, core::ptr::null_mut()) } != 0 {
                return Err(io::Error::UNKNOWN_THREAD_COUNT);
            }
            Ok(unsafe { NonZero::new_unchecked(cpus as usize) })
        } else if #[cfg(target_os = "haiku")] {
            // system_info cpu_count field gets the static data set at boot time with `smp_set_num_cpus`
            // `get_system_info` calls then `smp_get_num_cpus`
            unsafe {
                let mut sinfo: libc::system_info = crate::mem::zeroed();
                let res = libc::get_system_info(&mut sinfo);

                if res != libc::B_OK {
                    return Err(io::Error::UNKNOWN_THREAD_COUNT);
                }

                Ok(NonZero::new_unchecked(sinfo.cpu_count as usize))
            }
        } else if #[cfg(target_os = "vxworks")] {
            // Note: there is also `vxCpuConfiguredGet`, closer to _SC_NPROCESSORS_CONF
            // expectations than the actual cores availability.
            unsafe extern "C" {
                fn vxCpuEnabledGet() -> libc::cpuset_t;
            }

            // SAFETY: `vxCpuEnabledGet` always fetches a mask with at least one bit set
            unsafe{
                let set = vxCpuEnabledGet();
                Ok(NonZero::new_unchecked(set.count_ones() as usize))
            }
        } else {
            // FIXME: implement on Redox, l4re
            Err(io::const_error!(io::ErrorKind::Unsupported, "Getting the number of hardware threads is not supported on the target platform"))
        }
    }
}

#[cfg(any(target_os = "android", target_os = "linux"))]
mod cgroups {
    //! Currently not covered
    //! * cgroup v2 in non-standard mountpoints
    //! * paths containing control characters or spaces, since those would be escaped in procfs
    //!   output and we don't unescape

    use crate::borrow::Cow;
    use crate::ffi::OsString;
    use crate::fs::{File, exists};
    use crate::io::{BufRead, Read};
    use crate::os::unix::ffi::OsStringExt;
    use crate::path::{Path, PathBuf};
    use crate::str::from_utf8;

    #[derive(PartialEq)]
    enum Cgroup {
        V1,
        V2,
    }

    /// Returns cgroup CPU quota in core-equivalents, rounded down or usize::MAX if the quota cannot
    /// be determined or is not set.
    pub(super) fn quota() -> usize {
        let mut quota = usize::MAX;
        if cfg!(miri) {
            // Attempting to open a file fails under default flags due to isolation.
            // And Miri does not have parallelism anyway.
            return quota;
        }

        let _: Option<()> = try {
            let mut buf = Vec::with_capacity(128);
            // find our place in the cgroup hierarchy
            File::open("/proc/self/cgroup").ok()?.read_to_end(&mut buf).ok()?;
            let (cgroup_path, version) =
                buf.split(|&c| c == b'\n').fold(None, |previous, line| {
                    let mut fields = line.splitn(3, |&c| c == b':');
                    // 2nd field is a list of controllers for v1 or empty for v2
                    let version = match fields.nth(1) {
                        Some(b"") => Cgroup::V2,
                        Some(controllers)
                            if from_utf8(controllers)
                                .is_ok_and(|c| c.split(',').any(|c| c == "cpu")) =>
                        {
                            Cgroup::V1
                        }
                        _ => return previous,
                    };

                    // already-found v1 trumps v2 since it explicitly specifies its controllers
                    if previous.is_some() && version == Cgroup::V2 {
                        return previous;
                    }

                    let path = fields.last()?;
                    // skip leading slash
                    Some((path[1..].to_owned(), version))
                })?;
            let cgroup_path = PathBuf::from(OsString::from_vec(cgroup_path));

            quota = match version {
                Cgroup::V1 => quota_v1(cgroup_path),
                Cgroup::V2 => quota_v2(cgroup_path),
            };
        };

        quota
    }

    fn quota_v2(group_path: PathBuf) -> usize {
        let mut quota = usize::MAX;

        let mut path = PathBuf::with_capacity(128);
        let mut read_buf = String::with_capacity(20);

        // standard mount location defined in file-hierarchy(7) manpage
        let cgroup_mount = "/sys/fs/cgroup";

        path.push(cgroup_mount);
        path.push(&group_path);

        path.push("cgroup.controllers");

        // skip if we're not looking at cgroup2
        if matches!(exists(&path), Err(_) | Ok(false)) {
            return usize::MAX;
        };

        path.pop();

        let _: Option<()> = try {
            while path.starts_with(cgroup_mount) {
                path.push("cpu.max");

                read_buf.clear();

                if File::open(&path).and_then(|mut f| f.read_to_string(&mut read_buf)).is_ok() {
                    let raw_quota = read_buf.lines().next()?;
                    let mut raw_quota = raw_quota.split(' ');
                    let limit = raw_quota.next()?;
                    let period = raw_quota.next()?;
                    match (limit.parse::<usize>(), period.parse::<usize>()) {
                        (Ok(limit), Ok(period)) if period > 0 => {
                            quota = quota.min(limit / period);
                        }
                        _ => {}
                    }
                }

                path.pop(); // pop filename
                path.pop(); // pop dir
            }
        };

        quota
    }

    fn quota_v1(group_path: PathBuf) -> usize {
        let mut quota = usize::MAX;
        let mut path = PathBuf::with_capacity(128);
        let mut read_buf = String::with_capacity(20);

        // Hardcode commonly used locations mentioned in the cgroups(7) manpage
        // if that doesn't work scan mountinfo and adjust `group_path` for bind-mounts
        let mounts: &[fn(&Path) -> Option<(_, &Path)>] = &[
            |p| Some((Cow::Borrowed("/sys/fs/cgroup/cpu"), p)),
            |p| Some((Cow::Borrowed("/sys/fs/cgroup/cpu,cpuacct"), p)),
            // this can be expensive on systems with tons of mountpoints
            // but we only get to this point when /proc/self/cgroups explicitly indicated
            // this process belongs to a cpu-controller cgroup v1 and the defaults didn't work
            find_mountpoint,
        ];

        for mount in mounts {
            let Some((mount, group_path)) = mount(&group_path) else { continue };

            path.clear();
            path.push(mount.as_ref());
            path.push(&group_path);

            // skip if we guessed the mount incorrectly
            if matches!(exists(&path), Err(_) | Ok(false)) {
                continue;
            }

            while path.starts_with(mount.as_ref()) {
                let mut parse_file = |name| {
                    path.push(name);
                    read_buf.clear();

                    let f = File::open(&path);
                    path.pop(); // restore buffer before any early returns
                    f.ok()?.read_to_string(&mut read_buf).ok()?;
                    let parsed = read_buf.trim().parse::<usize>().ok()?;

                    Some(parsed)
                };

                let limit = parse_file("cpu.cfs_quota_us");
                let period = parse_file("cpu.cfs_period_us");

                match (limit, period) {
                    (Some(limit), Some(period)) if period > 0 => quota = quota.min(limit / period),
                    _ => {}
                }

                path.pop();
            }

            // we passed the try_exists above so we should have traversed the correct hierarchy
            // when reaching this line
            break;
        }

        quota
    }

    /// Scan mountinfo for cgroup v1 mountpoint with a cpu controller
    ///
    /// If the cgroupfs is a bind mount then `group_path` is adjusted to skip
    /// over the already-included prefix
    fn find_mountpoint(group_path: &Path) -> Option<(Cow<'static, str>, &Path)> {
        let mut reader = File::open_buffered("/proc/self/mountinfo").ok()?;
        let mut line = String::with_capacity(256);
        loop {
            line.clear();
            if reader.read_line(&mut line).ok()? == 0 {
                break;
            }

            let line = line.trim();
            let mut items = line.split(' ');

            let sub_path = items.nth(3)?;
            let mount_point = items.next()?;
            let mount_opts = items.next_back()?;
            let filesystem_type = items.nth_back(1)?;

            if filesystem_type != "cgroup" || !mount_opts.split(',').any(|opt| opt == "cpu") {
                // not a cgroup / not a cpu-controller
                continue;
            }

            let sub_path = Path::new(sub_path).strip_prefix("/").ok()?;

            if !group_path.starts_with(sub_path) {
                // this is a bind-mount and the bound subdirectory
                // does not contain the cgroup this process belongs to
                continue;
            }

            let trimmed_group_path = group_path.strip_prefix(sub_path).ok()?;

            return Some((Cow::Owned(mount_point.to_owned()), trimmed_group_path));
        }

        None
    }
}

// glibc >= 2.15 has a __pthread_get_minstack() function that returns
// PTHREAD_STACK_MIN plus bytes needed for thread-local storage.
// We need that information to avoid blowing up when a small stack
// is created in an application with big thread-local storage requirements.
// See #6233 for rationale and details.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
unsafe fn min_stack_size(attr: *const libc::pthread_attr_t) -> usize {
    // We use dlsym to avoid an ELF version dependency on GLIBC_PRIVATE. (#23628)
    // We shouldn't really be using such an internal symbol, but there's currently
    // no other way to account for the TLS size.
    dlsym!(fn __pthread_get_minstack(*const libc::pthread_attr_t) -> libc::size_t);

    match __pthread_get_minstack.get() {
        None => libc::PTHREAD_STACK_MIN,
        Some(f) => unsafe { f(attr) },
    }
}

// No point in looking up __pthread_get_minstack() on non-glibc platforms.
#[cfg(all(
    not(all(target_os = "linux", target_env = "gnu")),
    not(any(target_os = "netbsd", target_os = "nuttx"))
))]
unsafe fn min_stack_size(_: *const libc::pthread_attr_t) -> usize {
    libc::PTHREAD_STACK_MIN
}

#[cfg(any(target_os = "netbsd", target_os = "nuttx"))]
unsafe fn min_stack_size(_: *const libc::pthread_attr_t) -> usize {
    static STACK: crate::sync::OnceLock<usize> = crate::sync::OnceLock::new();

    *STACK.get_or_init(|| {
        let mut stack = unsafe { libc::sysconf(libc::_SC_THREAD_STACK_MIN) };
        if stack < 0 {
            stack = 2048; // just a guess
        }

        stack as usize
    })
}
