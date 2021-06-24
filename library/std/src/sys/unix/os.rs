//! Implementation of `std::os` functionality for unix systems

#![allow(unused_imports)] // lots of cfg code here

#[cfg(all(test, target_env = "gnu"))]
mod tests;

use crate::os::unix::prelude::*;

use crate::error::Error as StdError;
use crate::ffi::{CStr, CString, OsStr, OsString};
use crate::fmt;
use crate::io;
use crate::iter;
use crate::mem;
use crate::path::{self, PathBuf};
use crate::ptr;
use crate::slice;
use crate::str;
use crate::sys::cvt;
use crate::sys::fd;
use crate::sys::memchr;
use crate::sys_common::rwlock::{StaticRWLock, StaticRWLockReadGuard};
use crate::vec;

use libc::{c_char, c_int, c_void};

const TMPBUF_SZ: usize = 128;

cfg_if::cfg_if! {
    if #[cfg(target_os = "redox")] {
        const PATH_SEPARATOR: u8 = b';';
    } else {
        const PATH_SEPARATOR: u8 = b':';
    }
}

extern "C" {
    #[cfg(not(any(target_os = "dragonfly", target_os = "vxworks")))]
    #[cfg_attr(
        any(
            target_os = "linux",
            target_os = "emscripten",
            target_os = "fuchsia",
            target_os = "l4re"
        ),
        link_name = "__errno_location"
    )]
    #[cfg_attr(
        any(
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "android",
            target_os = "redox",
            target_env = "newlib"
        ),
        link_name = "__errno"
    )]
    #[cfg_attr(any(target_os = "solaris", target_os = "illumos"), link_name = "___errno")]
    #[cfg_attr(
        any(target_os = "macos", target_os = "ios", target_os = "freebsd"),
        link_name = "__error"
    )]
    #[cfg_attr(target_os = "haiku", link_name = "_errnop")]
    fn errno_location() -> *mut c_int;
}

/// Returns the platform-specific value of errno
#[cfg(not(any(target_os = "dragonfly", target_os = "vxworks")))]
pub fn errno() -> i32 {
    unsafe { (*errno_location()) as i32 }
}

/// Sets the platform-specific value of errno
#[cfg(all(not(target_os = "linux"), not(target_os = "dragonfly"), not(target_os = "vxworks")))] // needed for readdir and syscall!
#[allow(dead_code)] // but not all target cfgs actually end up using it
pub fn set_errno(e: i32) {
    unsafe { *errno_location() = e as c_int }
}

#[cfg(target_os = "vxworks")]
pub fn errno() -> i32 {
    unsafe { libc::errnoGet() }
}

#[cfg(target_os = "dragonfly")]
pub fn errno() -> i32 {
    extern "C" {
        #[thread_local]
        static errno: c_int;
    }

    unsafe { errno as i32 }
}

#[cfg(target_os = "dragonfly")]
pub fn set_errno(e: i32) {
    extern "C" {
        #[thread_local]
        static mut errno: c_int;
    }

    unsafe {
        errno = e;
    }
}

/// Gets a detailed string description for the given error number.
pub fn error_string(errno: i32) -> String {
    extern "C" {
        #[cfg_attr(any(target_os = "linux", target_env = "newlib"), link_name = "__xpg_strerror_r")]
        fn strerror_r(errnum: c_int, buf: *mut c_char, buflen: libc::size_t) -> c_int;
    }

    let mut buf = [0 as c_char; TMPBUF_SZ];

    let p = buf.as_mut_ptr();
    unsafe {
        if strerror_r(errno as c_int, p, buf.len()) < 0 {
            panic!("strerror_r failure");
        }

        let p = p as *const _;
        str::from_utf8(CStr::from_ptr(p).to_bytes()).unwrap().to_owned()
    }
}

pub fn getcwd() -> io::Result<PathBuf> {
    let mut buf = Vec::with_capacity(512);
    loop {
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut libc::c_char;
            if !libc::getcwd(ptr, buf.capacity()).is_null() {
                let len = CStr::from_ptr(buf.as_ptr() as *const libc::c_char).to_bytes().len();
                buf.set_len(len);
                buf.shrink_to_fit();
                return Ok(PathBuf::from(OsString::from_vec(buf)));
            } else {
                let error = io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ERANGE) {
                    return Err(error);
                }
            }

            // Trigger the internal buffer resizing logic of `Vec` by requiring
            // more space than the current capacity.
            let cap = buf.capacity();
            buf.set_len(cap);
            buf.reserve(1);
        }
    }
}

pub fn chdir(p: &path::Path) -> io::Result<()> {
    let p: &OsStr = p.as_ref();
    let p = CString::new(p.as_bytes())?;
    if unsafe { libc::chdir(p.as_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub struct SplitPaths<'a> {
    iter: iter::Map<slice::Split<'a, u8, fn(&u8) -> bool>, fn(&'a [u8]) -> PathBuf>,
}

pub fn split_paths(unparsed: &OsStr) -> SplitPaths<'_> {
    fn bytes_to_path(b: &[u8]) -> PathBuf {
        PathBuf::from(<OsStr as OsStrExt>::from_bytes(b))
    }
    fn is_separator(b: &u8) -> bool {
        *b == PATH_SEPARATOR
    }
    let unparsed = unparsed.as_bytes();
    SplitPaths {
        iter: unparsed
            .split(is_separator as fn(&u8) -> bool)
            .map(bytes_to_path as fn(&[u8]) -> PathBuf),
    }
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[derive(Debug)]
pub struct JoinPathsError;

pub fn join_paths<I, T>(paths: I) -> Result<OsString, JoinPathsError>
where
    I: Iterator<Item = T>,
    T: AsRef<OsStr>,
{
    let mut joined = Vec::new();

    for (i, path) in paths.enumerate() {
        let path = path.as_ref().as_bytes();
        if i > 0 {
            joined.push(PATH_SEPARATOR)
        }
        if path.contains(&PATH_SEPARATOR) {
            return Err(JoinPathsError);
        }
        joined.extend_from_slice(path);
    }
    Ok(OsStringExt::from_vec(joined))
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "path segment contains separator `{}`", char::from(PATH_SEPARATOR))
    }
}

impl StdError for JoinPathsError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "failed to join paths"
    }
}

#[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
pub fn current_exe() -> io::Result<PathBuf> {
    unsafe {
        let mut mib = [
            libc::CTL_KERN as c_int,
            libc::KERN_PROC as c_int,
            libc::KERN_PROC_PATHNAME as c_int,
            -1 as c_int,
        ];
        let mut sz = 0;
        cvt(libc::sysctl(
            mib.as_mut_ptr(),
            mib.len() as libc::c_uint,
            ptr::null_mut(),
            &mut sz,
            ptr::null_mut(),
            0,
        ))?;
        if sz == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut v: Vec<u8> = Vec::with_capacity(sz);
        cvt(libc::sysctl(
            mib.as_mut_ptr(),
            mib.len() as libc::c_uint,
            v.as_mut_ptr() as *mut libc::c_void,
            &mut sz,
            ptr::null_mut(),
            0,
        ))?;
        if sz == 0 {
            return Err(io::Error::last_os_error());
        }
        v.set_len(sz - 1); // chop off trailing NUL
        Ok(PathBuf::from(OsString::from_vec(v)))
    }
}

#[cfg(target_os = "netbsd")]
pub fn current_exe() -> io::Result<PathBuf> {
    fn sysctl() -> io::Result<PathBuf> {
        unsafe {
            let mib = [libc::CTL_KERN, libc::KERN_PROC_ARGS, -1, libc::KERN_PROC_PATHNAME];
            let mut path_len: usize = 0;
            cvt(libc::sysctl(
                mib.as_ptr(),
                mib.len() as libc::c_uint,
                ptr::null_mut(),
                &mut path_len,
                ptr::null(),
                0,
            ))?;
            if path_len <= 1 {
                return Err(io::Error::new_const(
                    io::ErrorKind::Uncategorized,
                    &"KERN_PROC_PATHNAME sysctl returned zero-length string",
                ));
            }
            let mut path: Vec<u8> = Vec::with_capacity(path_len);
            cvt(libc::sysctl(
                mib.as_ptr(),
                mib.len() as libc::c_uint,
                path.as_ptr() as *mut libc::c_void,
                &mut path_len,
                ptr::null(),
                0,
            ))?;
            path.set_len(path_len - 1); // chop off NUL
            Ok(PathBuf::from(OsString::from_vec(path)))
        }
    }
    fn procfs() -> io::Result<PathBuf> {
        let curproc_exe = path::Path::new("/proc/curproc/exe");
        if curproc_exe.is_file() {
            return crate::fs::read_link(curproc_exe);
        }
        Err(io::Error::new_const(
            io::ErrorKind::Uncategorized,
            &"/proc/curproc/exe doesn't point to regular file.",
        ))
    }
    sysctl().or_else(|_| procfs())
}

#[cfg(target_os = "openbsd")]
pub fn current_exe() -> io::Result<PathBuf> {
    unsafe {
        let mut mib = [libc::CTL_KERN, libc::KERN_PROC_ARGS, libc::getpid(), libc::KERN_PROC_ARGV];
        let mib = mib.as_mut_ptr();
        let mut argv_len = 0;
        cvt(libc::sysctl(mib, 4, ptr::null_mut(), &mut argv_len, ptr::null_mut(), 0))?;
        let mut argv = Vec::<*const libc::c_char>::with_capacity(argv_len as usize);
        cvt(libc::sysctl(mib, 4, argv.as_mut_ptr() as *mut _, &mut argv_len, ptr::null_mut(), 0))?;
        argv.set_len(argv_len as usize);
        if argv[0].is_null() {
            return Err(io::Error::new_const(
                io::ErrorKind::Uncategorized,
                &"no current exe available",
            ));
        }
        let argv0 = CStr::from_ptr(argv[0]).to_bytes();
        if argv0[0] == b'.' || argv0.iter().any(|b| *b == b'/') {
            crate::fs::canonicalize(OsStr::from_bytes(argv0))
        } else {
            Ok(PathBuf::from(OsStr::from_bytes(argv0)))
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "emscripten"))]
pub fn current_exe() -> io::Result<PathBuf> {
    match crate::fs::read_link("/proc/self/exe") {
        Err(ref e) if e.kind() == io::ErrorKind::NotFound => Err(io::Error::new_const(
            io::ErrorKind::Uncategorized,
            &"no /proc/self/exe available. Is /proc mounted?",
        )),
        other => other,
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn current_exe() -> io::Result<PathBuf> {
    extern "C" {
        fn _NSGetExecutablePath(buf: *mut libc::c_char, bufsize: *mut u32) -> libc::c_int;
    }
    unsafe {
        let mut sz: u32 = 0;
        _NSGetExecutablePath(ptr::null_mut(), &mut sz);
        if sz == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut v: Vec<u8> = Vec::with_capacity(sz as usize);
        let err = _NSGetExecutablePath(v.as_mut_ptr() as *mut i8, &mut sz);
        if err != 0 {
            return Err(io::Error::last_os_error());
        }
        v.set_len(sz as usize - 1); // chop off trailing NUL
        Ok(PathBuf::from(OsString::from_vec(v)))
    }
}

#[cfg(any(target_os = "solaris", target_os = "illumos"))]
pub fn current_exe() -> io::Result<PathBuf> {
    extern "C" {
        fn getexecname() -> *const c_char;
    }
    unsafe {
        let path = getexecname();
        if path.is_null() {
            Err(io::Error::last_os_error())
        } else {
            let filename = CStr::from_ptr(path).to_bytes();
            let path = PathBuf::from(<OsStr as OsStrExt>::from_bytes(filename));

            // Prepend a current working directory to the path if
            // it doesn't contain an absolute pathname.
            if filename[0] == b'/' { Ok(path) } else { getcwd().map(|cwd| cwd.join(path)) }
        }
    }
}

#[cfg(target_os = "haiku")]
pub fn current_exe() -> io::Result<PathBuf> {
    // Use Haiku's image info functions
    #[repr(C)]
    struct image_info {
        id: i32,
        type_: i32,
        sequence: i32,
        init_order: i32,
        init_routine: *mut libc::c_void, // function pointer
        term_routine: *mut libc::c_void, // function pointer
        device: libc::dev_t,
        node: libc::ino_t,
        name: [libc::c_char; 1024], // MAXPATHLEN
        text: *mut libc::c_void,
        data: *mut libc::c_void,
        text_size: i32,
        data_size: i32,
        api_version: i32,
        abi: i32,
    }

    unsafe {
        extern "C" {
            fn _get_next_image_info(
                team_id: i32,
                cookie: *mut i32,
                info: *mut image_info,
                size: i32,
            ) -> i32;
        }

        let mut info: image_info = mem::zeroed();
        let mut cookie: i32 = 0;
        // the executable can be found at team id 0
        let result =
            _get_next_image_info(0, &mut cookie, &mut info, mem::size_of::<image_info>() as i32);
        if result != 0 {
            use crate::io::ErrorKind;
            Err(io::Error::new_const(ErrorKind::Uncategorized, &"Error getting executable path"))
        } else {
            let name = CStr::from_ptr(info.name.as_ptr()).to_bytes();
            Ok(PathBuf::from(OsStr::from_bytes(name)))
        }
    }
}

#[cfg(target_os = "redox")]
pub fn current_exe() -> io::Result<PathBuf> {
    crate::fs::read_to_string("sys:exe").map(PathBuf::from)
}

#[cfg(any(target_os = "fuchsia", target_os = "l4re"))]
pub fn current_exe() -> io::Result<PathBuf> {
    use crate::io::ErrorKind;
    Err(io::Error::new_const(ErrorKind::Unsupported, &"Not yet implemented!"))
}

#[cfg(target_os = "vxworks")]
pub fn current_exe() -> io::Result<PathBuf> {
    #[cfg(test)]
    use realstd::env;

    #[cfg(not(test))]
    use crate::env;

    let exe_path = env::args().next().unwrap();
    let path = path::Path::new(&exe_path);
    path.canonicalize()
}

pub struct Env {
    iter: vec::IntoIter<(OsString, OsString)>,
}

impl !Send for Env {}
impl !Sync for Env {}

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[cfg(target_os = "macos")]
pub unsafe fn environ() -> *mut *const *const c_char {
    extern "C" {
        fn _NSGetEnviron() -> *mut *const *const c_char;
    }
    _NSGetEnviron()
}

#[cfg(not(target_os = "macos"))]
pub unsafe fn environ() -> *mut *const *const c_char {
    extern "C" {
        static mut environ: *const *const c_char;
    }
    ptr::addr_of_mut!(environ)
}

static ENV_LOCK: StaticRWLock = StaticRWLock::new();

pub fn env_read_lock() -> StaticRWLockReadGuard {
    ENV_LOCK.read()
}

/// Returns a vector of (variable, value) byte-vector pairs for all the
/// environment variables of the current process.
pub fn env() -> Env {
    unsafe {
        let _guard = env_read_lock();
        let mut environ = *environ();
        let mut result = Vec::new();
        if !environ.is_null() {
            while !(*environ).is_null() {
                if let Some(key_value) = parse(CStr::from_ptr(*environ).to_bytes()) {
                    result.push(key_value);
                }
                environ = environ.add(1);
            }
        }
        return Env { iter: result.into_iter() };
    }

    fn parse(input: &[u8]) -> Option<(OsString, OsString)> {
        // Strategy (copied from glibc): Variable name and value are separated
        // by an ASCII equals sign '='. Since a variable name must not be
        // empty, allow variable names starting with an equals sign. Skip all
        // malformed lines.
        if input.is_empty() {
            return None;
        }
        let pos = memchr::memchr(b'=', &input[1..]).map(|p| p + 1);
        pos.map(|p| {
            (
                OsStringExt::from_vec(input[..p].to_vec()),
                OsStringExt::from_vec(input[p + 1..].to_vec()),
            )
        })
    }
}

pub fn getenv(k: &OsStr) -> Option<OsString> {
    // environment variables with a nul byte can't be set, so their value is
    // always None as well
    let k = CString::new(k.as_bytes()).ok()?;
    unsafe {
        let _guard = env_read_lock();
        let s = libc::getenv(k.as_ptr()) as *const libc::c_char;
        if s.is_null() {
            None
        } else {
            Some(OsStringExt::from_vec(CStr::from_ptr(s).to_bytes().to_vec()))
        }
    }
}

pub fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    let k = CString::new(k.as_bytes())?;
    let v = CString::new(v.as_bytes())?;

    unsafe {
        let _guard = ENV_LOCK.write();
        cvt(libc::setenv(k.as_ptr(), v.as_ptr(), 1)).map(drop)
    }
}

pub fn unsetenv(n: &OsStr) -> io::Result<()> {
    let nbuf = CString::new(n.as_bytes())?;

    unsafe {
        let _guard = ENV_LOCK.write();
        cvt(libc::unsetenv(nbuf.as_ptr())).map(drop)
    }
}

pub fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

pub fn temp_dir() -> PathBuf {
    crate::env::var_os("TMPDIR").map(PathBuf::from).unwrap_or_else(|| {
        if cfg!(target_os = "android") {
            PathBuf::from("/data/local/tmp")
        } else {
            PathBuf::from("/tmp")
        }
    })
}

pub fn home_dir() -> Option<PathBuf> {
    return crate::env::var_os("HOME").or_else(|| unsafe { fallback() }).map(PathBuf::from);

    #[cfg(any(
        target_os = "android",
        target_os = "ios",
        target_os = "emscripten",
        target_os = "redox",
        target_os = "vxworks"
    ))]
    unsafe fn fallback() -> Option<OsString> {
        None
    }
    #[cfg(not(any(
        target_os = "android",
        target_os = "ios",
        target_os = "emscripten",
        target_os = "redox",
        target_os = "vxworks"
    )))]
    unsafe fn fallback() -> Option<OsString> {
        let amt = match libc::sysconf(libc::_SC_GETPW_R_SIZE_MAX) {
            n if n < 0 => 512 as usize,
            n => n as usize,
        };
        let mut buf = Vec::with_capacity(amt);
        let mut passwd: libc::passwd = mem::zeroed();
        let mut result = ptr::null_mut();
        match libc::getpwuid_r(
            libc::getuid(),
            &mut passwd,
            buf.as_mut_ptr(),
            buf.capacity(),
            &mut result,
        ) {
            0 if !result.is_null() => {
                let ptr = passwd.pw_dir as *const _;
                let bytes = CStr::from_ptr(ptr).to_bytes().to_vec();
                Some(OsStringExt::from_vec(bytes))
            }
            _ => None,
        }
    }
}

pub fn exit(code: i32) -> ! {
    unsafe { libc::exit(code as c_int) }
}

pub fn getpid() -> u32 {
    unsafe { libc::getpid() as u32 }
}

pub fn getppid() -> u32 {
    unsafe { libc::getppid() as u32 }
}

#[cfg(all(target_env = "gnu", not(target_os = "vxworks")))]
pub fn glibc_version() -> Option<(usize, usize)> {
    if let Some(Ok(version_str)) = glibc_version_cstr().map(CStr::to_str) {
        parse_glibc_version(version_str)
    } else {
        None
    }
}

#[cfg(all(target_env = "gnu", not(target_os = "vxworks")))]
fn glibc_version_cstr() -> Option<&'static CStr> {
    weak! {
        fn gnu_get_libc_version() -> *const libc::c_char
    }
    if let Some(f) = gnu_get_libc_version.get() {
        unsafe { Some(CStr::from_ptr(f())) }
    } else {
        None
    }
}

// Returns Some((major, minor)) if the string is a valid "x.y" version,
// ignoring any extra dot-separated parts. Otherwise return None.
#[cfg(all(target_env = "gnu", not(target_os = "vxworks")))]
fn parse_glibc_version(version: &str) -> Option<(usize, usize)> {
    let mut parsed_ints = version.split('.').map(str::parse::<usize>).fuse();
    match (parsed_ints.next(), parsed_ints.next()) {
        (Some(Ok(major)), Some(Ok(minor))) => Some((major, minor)),
        _ => None,
    }
}
