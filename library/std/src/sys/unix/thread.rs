use crate::cmp;
use crate::ffi::CStr;
use crate::io;
use crate::mem;
use crate::num::NonZeroUsize;
use crate::ptr;
use crate::sys::{os, stack_overflow};
use crate::time::Duration;

#[cfg(all(target_os = "linux", target_env = "gnu"))]
use crate::sys::weak::dlsym;
#[cfg(any(target_os = "solaris", target_os = "illumos", target_os = "nto"))]
use crate::sys::weak::weak;
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

    extern "C" {
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
    pub unsafe fn new(stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        let p = Box::into_raw(Box::new(p));
        let mut native: libc::pthread_t = mem::zeroed();
        let mut attr: libc::pthread_attr_t = mem::zeroed();
        assert_eq!(libc::pthread_attr_init(&mut attr), 0);

        #[cfg(target_os = "espidf")]
        if stack > 0 {
            // Only set the stack if a non-zero value is passed
            // 0 is used as an indication that the default stack size configured in the ESP-IDF menuconfig system should be used
            assert_eq!(
                libc::pthread_attr_setstacksize(&mut attr, cmp::max(stack, min_stack_size(&attr))),
                0
            );
        }

        #[cfg(not(target_os = "espidf"))]
        {
            let stack_size = cmp::max(stack, min_stack_size(&attr));

            match libc::pthread_attr_setstacksize(&mut attr, stack_size) {
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
                    assert_eq!(libc::pthread_attr_setstacksize(&mut attr, stack_size), 0);
                }
            };
        }

        let ret = libc::pthread_create(&mut native, &attr, thread_start, p as *mut _);
        // Note: if the thread creation fails and this assert fails, then p will
        // be leaked. However, an alternative design could cause double-free
        // which is clearly worse.
        assert_eq!(libc::pthread_attr_destroy(&mut attr), 0);

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
            libc::prctl(
                PR_SET_NAME,
                name.as_ptr(),
                0 as libc::c_ulong,
                0 as libc::c_ulong,
                0 as libc::c_ulong,
            );
        }
    }

    #[cfg(target_os = "linux")]
    pub fn set_name(name: &CStr) {
        const TASK_COMM_LEN: usize = 16;

        unsafe {
            // Available since glibc 2.12, musl 1.1.16, and uClibc 1.0.20.
            let name = truncate_cstr::<{ TASK_COMM_LEN }>(name);
            let res = libc::pthread_setname_np(libc::pthread_self(), name.as_ptr());
            // We have no good way of propagating errors here, but in debug-builds let's check that this actually worked.
            debug_assert_eq!(res, 0);
        }
    }

    #[cfg(any(target_os = "freebsd", target_os = "dragonfly", target_os = "openbsd"))]
    pub fn set_name(name: &CStr) {
        unsafe {
            libc::pthread_set_name_np(libc::pthread_self(), name.as_ptr());
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "watchos"))]
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
            let cname = CStr::from_bytes_with_nul_unchecked(b"%s\0".as_slice());
            let res = libc::pthread_setname_np(
                libc::pthread_self(),
                cname.as_ptr(),
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
            libc::rename_thread(thread_self, name.as_ptr());
        }
    }

    #[cfg(any(
        target_env = "newlib",
        target_os = "l4re",
        target_os = "emscripten",
        target_os = "redox",
        target_os = "vxworks"
    ))]
    pub fn set_name(_name: &CStr) {
        // Newlib, Emscripten, and VxWorks have no way to set a thread name.
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
                let ts_ptr = &mut ts as *mut _;
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
        let mut micros = dur.as_micros();
        unsafe {
            while micros > 0 {
                let st = if micros > u32::MAX as u128 { u32::MAX } else { micros as u32 };
                libc::usleep(st);

                micros -= st as u128;
            }
        }
    }

    pub fn join(self) {
        unsafe {
            let ret = libc::pthread_join(self.id, ptr::null_mut());
            mem::forget(self);
            assert!(ret == 0, "failed to join thread: {}", io::Error::from_raw_os_error(ret));
        }
    }

    pub fn id(&self) -> libc::pthread_t {
        self.id
    }

    pub fn into_id(self) -> libc::pthread_t {
        let id = self.id;
        mem::forget(self);
        id
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
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
))]
fn truncate_cstr<const MAX_WITH_NUL: usize>(cstr: &CStr) -> [libc::c_char; MAX_WITH_NUL] {
    let mut result = [0; MAX_WITH_NUL];
    for (src, dst) in cstr.to_bytes().iter().zip(&mut result[..MAX_WITH_NUL - 1]) {
        *dst = *src as libc::c_char;
    }
    result
}

pub fn available_parallelism() -> io::Result<NonZeroUsize> {
    cfg_if::cfg_if! {
        if #[cfg(any(
            target_os = "android",
            target_os = "emscripten",
            target_os = "fuchsia",
            target_os = "ios",
            target_os = "tvos",
            target_os = "linux",
            target_os = "macos",
            target_os = "solaris",
            target_os = "illumos",
        ))] {
            #[cfg(any(target_os = "android", target_os = "linux"))]
            {
                let quota = cgroups::quota().max(1);
                let mut set: libc::cpu_set_t = unsafe { mem::zeroed() };
                unsafe {
                    if libc::sched_getaffinity(0, mem::size_of::<libc::cpu_set_t>(), &mut set) == 0 {
                        let count = libc::CPU_COUNT(&set) as usize;
                        let count = count.min(quota);
                        // SAFETY: affinity mask can't be empty and the quota gets clamped to a minimum of 1
                        return Ok(NonZeroUsize::new_unchecked(count));
                    }
                }
            }
            match unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) } {
                -1 => Err(io::Error::last_os_error()),
                0 => Err(io::const_io_error!(io::ErrorKind::NotFound, "The number of hardware threads is not known for the target platform")),
                cpus => Ok(unsafe { NonZeroUsize::new_unchecked(cpus as usize) }),
            }
        } else if #[cfg(any(target_os = "freebsd", target_os = "dragonfly", target_os = "netbsd"))] {
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
                            return Ok(NonZeroUsize::new_unchecked(count));
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
                            for i in 0..u64::MAX {
                                match libc::_cpuset_isset(i, set) {
                                    -1 => break,
                                    0 => continue,
                                    _ => count = count + 1,
                                }
                            }
                        }
                        libc::_cpuset_destroy(set);
                        if let Some(count) = NonZeroUsize::new(count) {
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
                        &mut cpus as *mut _ as *mut _,
                        &mut cpus_size as *mut _ as *mut _,
                        ptr::null_mut(),
                        0,
                    )
                };

                // Handle errors if any.
                if res == -1 {
                    return Err(io::Error::last_os_error());
                } else if cpus == 0 {
                    return Err(io::const_io_error!(io::ErrorKind::NotFound, "The number of hardware threads is not known for the target platform"));
                }
            }
            Ok(unsafe { NonZeroUsize::new_unchecked(cpus as usize) })
        } else if #[cfg(target_os = "openbsd")] {
            use crate::ptr;

            let mut cpus: libc::c_uint = 0;
            let mut cpus_size = crate::mem::size_of_val(&cpus);
            let mut mib = [libc::CTL_HW, libc::HW_NCPU, 0, 0];

            let res = unsafe {
                libc::sysctl(
                    mib.as_mut_ptr(),
                    2,
                    &mut cpus as *mut _ as *mut _,
                    &mut cpus_size as *mut _ as *mut _,
                    ptr::null_mut(),
                    0,
                )
            };

            // Handle errors if any.
            if res == -1 {
                return Err(io::Error::last_os_error());
            } else if cpus == 0 {
                return Err(io::const_io_error!(io::ErrorKind::NotFound, "The number of hardware threads is not known for the target platform"));
            }

            Ok(unsafe { NonZeroUsize::new_unchecked(cpus as usize) })
        } else if #[cfg(target_os = "nto")] {
            unsafe {
                use libc::_syspage_ptr;
                if _syspage_ptr.is_null() {
                    Err(io::const_io_error!(io::ErrorKind::NotFound, "No syspage available"))
                } else {
                    let cpus = (*_syspage_ptr).num_cpu;
                    NonZeroUsize::new(cpus as usize)
                        .ok_or(io::const_io_error!(io::ErrorKind::NotFound, "The number of hardware threads is not known for the target platform"))
                }
            }
        } else if #[cfg(target_os = "haiku")] {
            // system_info cpu_count field gets the static data set at boot time with `smp_set_num_cpus`
            // `get_system_info` calls then `smp_get_num_cpus`
            unsafe {
                let mut sinfo: libc::system_info = crate::mem::zeroed();
                let res = libc::get_system_info(&mut sinfo);

                if res != libc::B_OK {
                    return Err(io::const_io_error!(io::ErrorKind::NotFound, "The number of hardware threads is not known for the target platform"));
                }

                Ok(NonZeroUsize::new_unchecked(sinfo.cpu_count as usize))
            }
        } else {
            // FIXME: implement on vxWorks, Redox, l4re
            Err(io::const_io_error!(io::ErrorKind::Unsupported, "Getting the number of hardware threads is not supported on the target platform"))
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
    use crate::fs::{try_exists, File};
    use crate::io::Read;
    use crate::io::{BufRead, BufReader};
    use crate::os::unix::ffi::OsStringExt;
    use crate::path::Path;
    use crate::path::PathBuf;
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
        if matches!(try_exists(&path), Err(_) | Ok(false)) {
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
            if matches!(try_exists(&path), Err(_) | Ok(false)) {
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
        let mut reader = BufReader::new(File::open("/proc/self/mountinfo").ok()?);
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

#[cfg(all(
    not(target_os = "linux"),
    not(target_os = "freebsd"),
    not(target_os = "macos"),
    not(target_os = "netbsd"),
    not(target_os = "openbsd"),
    not(target_os = "solaris")
))]
#[cfg_attr(test, allow(dead_code))]
pub mod guard {
    use crate::ops::Range;
    pub type Guard = Range<usize>;
    pub unsafe fn current() -> Option<Guard> {
        None
    }
    pub unsafe fn init() -> Option<Guard> {
        None
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "freebsd",
    target_os = "macos",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "solaris"
))]
#[cfg_attr(test, allow(dead_code))]
pub mod guard {
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    use libc::{mmap as mmap64, mprotect};
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    use libc::{mmap64, mprotect};
    use libc::{MAP_ANON, MAP_FAILED, MAP_FIXED, MAP_PRIVATE, PROT_NONE, PROT_READ, PROT_WRITE};

    use crate::io;
    use crate::ops::Range;
    use crate::sync::atomic::{AtomicUsize, Ordering};
    use crate::sys::os;

    // This is initialized in init() and only read from after
    static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);

    pub type Guard = Range<usize>;

    #[cfg(target_os = "solaris")]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let mut current_stack: libc::stack_t = crate::mem::zeroed();
        assert_eq!(libc::stack_getbounds(&mut current_stack), 0);
        Some(current_stack.ss_sp)
    }

    #[cfg(target_os = "macos")]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let th = libc::pthread_self();
        let stackptr = libc::pthread_get_stackaddr_np(th);
        Some(stackptr.map_addr(|addr| addr - libc::pthread_get_stacksize_np(th)))
    }

    #[cfg(target_os = "openbsd")]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let mut current_stack: libc::stack_t = crate::mem::zeroed();
        assert_eq!(libc::pthread_stackseg_np(libc::pthread_self(), &mut current_stack), 0);

        let stack_ptr = current_stack.ss_sp;
        let stackaddr = if libc::pthread_main_np() == 1 {
            // main thread
            stack_ptr.addr() - current_stack.ss_size + PAGE_SIZE.load(Ordering::Relaxed)
        } else {
            // new thread
            stack_ptr.addr() - current_stack.ss_size
        };
        Some(stack_ptr.with_addr(stackaddr))
    }

    #[cfg(any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "l4re"
    ))]
    unsafe fn get_stack_start() -> Option<*mut libc::c_void> {
        let mut ret = None;
        let mut attr: libc::pthread_attr_t = crate::mem::zeroed();
        #[cfg(target_os = "freebsd")]
        assert_eq!(libc::pthread_attr_init(&mut attr), 0);
        #[cfg(target_os = "freebsd")]
        let e = libc::pthread_attr_get_np(libc::pthread_self(), &mut attr);
        #[cfg(not(target_os = "freebsd"))]
        let e = libc::pthread_getattr_np(libc::pthread_self(), &mut attr);
        if e == 0 {
            let mut stackaddr = crate::ptr::null_mut();
            let mut stacksize = 0;
            assert_eq!(libc::pthread_attr_getstack(&attr, &mut stackaddr, &mut stacksize), 0);
            ret = Some(stackaddr);
        }
        if e == 0 || cfg!(target_os = "freebsd") {
            assert_eq!(libc::pthread_attr_destroy(&mut attr), 0);
        }
        ret
    }

    // Precondition: PAGE_SIZE is initialized.
    unsafe fn get_stack_start_aligned() -> Option<*mut libc::c_void> {
        let page_size = PAGE_SIZE.load(Ordering::Relaxed);
        assert!(page_size != 0);
        let stackptr = get_stack_start()?;
        let stackaddr = stackptr.addr();

        // Ensure stackaddr is page aligned! A parent process might
        // have reset RLIMIT_STACK to be non-page aligned. The
        // pthread_attr_getstack() reports the usable stack area
        // stackaddr < stackaddr + stacksize, so if stackaddr is not
        // page-aligned, calculate the fix such that stackaddr <
        // new_page_aligned_stackaddr < stackaddr + stacksize
        let remainder = stackaddr % page_size;
        Some(if remainder == 0 {
            stackptr
        } else {
            stackptr.with_addr(stackaddr + page_size - remainder)
        })
    }

    pub unsafe fn init() -> Option<Guard> {
        let page_size = os::page_size();
        PAGE_SIZE.store(page_size, Ordering::Relaxed);

        if cfg!(all(target_os = "linux", not(target_env = "musl"))) {
            // Linux doesn't allocate the whole stack right away, and
            // the kernel has its own stack-guard mechanism to fault
            // when growing too close to an existing mapping. If we map
            // our own guard, then the kernel starts enforcing a rather
            // large gap above that, rendering much of the possible
            // stack space useless. See #43052.
            //
            // Instead, we'll just note where we expect rlimit to start
            // faulting, so our handler can report "stack overflow", and
            // trust that the kernel's own stack guard will work.
            let stackptr = get_stack_start_aligned()?;
            let stackaddr = stackptr.addr();
            Some(stackaddr - page_size..stackaddr)
        } else if cfg!(all(target_os = "linux", target_env = "musl")) {
            // For the main thread, the musl's pthread_attr_getstack
            // returns the current stack size, rather than maximum size
            // it can eventually grow to. It cannot be used to determine
            // the position of kernel's stack guard.
            None
        } else if cfg!(target_os = "freebsd") {
            // FreeBSD's stack autogrows, and optionally includes a guard page
            // at the bottom. If we try to remap the bottom of the stack
            // ourselves, FreeBSD's guard page moves upwards. So we'll just use
            // the builtin guard page.
            let stackptr = get_stack_start_aligned()?;
            let guardaddr = stackptr.addr();
            // Technically the number of guard pages is tunable and controlled
            // by the security.bsd.stack_guard_page sysctl, but there are
            // few reasons to change it from the default. The default value has
            // been 1 ever since FreeBSD 11.1 and 10.4.
            const GUARD_PAGES: usize = 1;
            let guard = guardaddr..guardaddr + GUARD_PAGES * page_size;
            Some(guard)
        } else if cfg!(target_os = "openbsd") {
            // OpenBSD stack already includes a guard page, and stack is
            // immutable.
            //
            // We'll just note where we expect rlimit to start
            // faulting, so our handler can report "stack overflow", and
            // trust that the kernel's own stack guard will work.
            let stackptr = get_stack_start_aligned()?;
            let stackaddr = stackptr.addr();
            Some(stackaddr - page_size..stackaddr)
        } else {
            // Reallocate the last page of the stack.
            // This ensures SIGBUS will be raised on
            // stack overflow.
            // Systems which enforce strict PAX MPROTECT do not allow
            // to mprotect() a mapping with less restrictive permissions
            // than the initial mmap() used, so we mmap() here with
            // read/write permissions and only then mprotect() it to
            // no permissions at all. See issue #50313.
            let stackptr = get_stack_start_aligned()?;
            let result = mmap64(
                stackptr,
                page_size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON | MAP_FIXED,
                -1,
                0,
            );
            if result != stackptr || result == MAP_FAILED {
                panic!("failed to allocate a guard page: {}", io::Error::last_os_error());
            }

            let result = mprotect(stackptr, page_size, PROT_NONE);
            if result != 0 {
                panic!("failed to protect the guard page: {}", io::Error::last_os_error());
            }

            let guardaddr = stackptr.addr();

            Some(guardaddr..guardaddr + page_size)
        }
    }

    #[cfg(any(target_os = "macos", target_os = "openbsd", target_os = "solaris"))]
    pub unsafe fn current() -> Option<Guard> {
        let stackptr = get_stack_start()?;
        let stackaddr = stackptr.addr();
        Some(stackaddr - PAGE_SIZE.load(Ordering::Relaxed)..stackaddr)
    }

    #[cfg(any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "netbsd",
        target_os = "l4re"
    ))]
    pub unsafe fn current() -> Option<Guard> {
        let mut ret = None;
        let mut attr: libc::pthread_attr_t = crate::mem::zeroed();
        #[cfg(target_os = "freebsd")]
        assert_eq!(libc::pthread_attr_init(&mut attr), 0);
        #[cfg(target_os = "freebsd")]
        let e = libc::pthread_attr_get_np(libc::pthread_self(), &mut attr);
        #[cfg(not(target_os = "freebsd"))]
        let e = libc::pthread_getattr_np(libc::pthread_self(), &mut attr);
        if e == 0 {
            let mut guardsize = 0;
            assert_eq!(libc::pthread_attr_getguardsize(&attr, &mut guardsize), 0);
            if guardsize == 0 {
                if cfg!(all(target_os = "linux", target_env = "musl")) {
                    // musl versions before 1.1.19 always reported guard
                    // size obtained from pthread_attr_get_np as zero.
                    // Use page size as a fallback.
                    guardsize = PAGE_SIZE.load(Ordering::Relaxed);
                } else {
                    panic!("there is no guard page");
                }
            }
            let mut stackptr = crate::ptr::null_mut::<libc::c_void>();
            let mut size = 0;
            assert_eq!(libc::pthread_attr_getstack(&attr, &mut stackptr, &mut size), 0);

            let stackaddr = stackptr.addr();
            ret = if cfg!(any(target_os = "freebsd", target_os = "netbsd")) {
                Some(stackaddr - guardsize..stackaddr)
            } else if cfg!(all(target_os = "linux", target_env = "musl")) {
                Some(stackaddr - guardsize..stackaddr)
            } else if cfg!(all(target_os = "linux", any(target_env = "gnu", target_env = "uclibc")))
            {
                // glibc used to include the guard area within the stack, as noted in the BUGS
                // section of `man pthread_attr_getguardsize`. This has been corrected starting
                // with glibc 2.27, and in some distro backports, so the guard is now placed at the
                // end (below) the stack. There's no easy way for us to know which we have at
                // runtime, so we'll just match any fault in the range right above or below the
                // stack base to call that fault a stack overflow.
                Some(stackaddr - guardsize..stackaddr + guardsize)
            } else {
                Some(stackaddr..stackaddr + guardsize)
            };
        }
        if e == 0 || cfg!(target_os = "freebsd") {
            assert_eq!(libc::pthread_attr_destroy(&mut attr), 0);
        }
        ret
    }
}

// glibc >= 2.15 has a __pthread_get_minstack() function that returns
// PTHREAD_STACK_MIN plus bytes needed for thread-local storage.
// We need that information to avoid blowing up when a small stack
// is created in an application with big thread-local storage requirements.
// See #6233 for rationale and details.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn min_stack_size(attr: *const libc::pthread_attr_t) -> usize {
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
#[cfg(all(not(all(target_os = "linux", target_env = "gnu")), not(target_os = "netbsd")))]
fn min_stack_size(_: *const libc::pthread_attr_t) -> usize {
    libc::PTHREAD_STACK_MIN
}

#[cfg(target_os = "netbsd")]
fn min_stack_size(_: *const libc::pthread_attr_t) -> usize {
    2048 // just a guess
}
