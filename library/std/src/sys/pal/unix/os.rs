//! Implementation of `std::os` functionality for unix systems

#![allow(unused_imports)] // lots of cfg code here

#[cfg(test)]
mod tests;

use core::slice::memchr;

use libc::{c_char, c_int, c_void};

use crate::error::Error as StdError;
use crate::ffi::{CStr, CString, OsStr, OsString};
use crate::os::unix::prelude::*;
use crate::path::{self, PathBuf};
use crate::sync::{PoisonError, RwLock};
use crate::sys::common::small_c_string::{run_path_with_cstr, run_with_cstr};
#[cfg(all(target_env = "gnu", not(target_os = "vxworks")))]
use crate::sys::weak::weak;
use crate::sys::{cvt, fd};
use crate::{fmt, io, iter, mem, ptr, slice, str, vec};

const TMPBUF_SZ: usize = 128;

cfg_if::cfg_if! {
    if #[cfg(target_os = "redox")] {
        const PATH_SEPARATOR: u8 = b';';
    } else {
        const PATH_SEPARATOR: u8 = b':';
    }
}

unsafe extern "C" {
    #[cfg(not(any(target_os = "dragonfly", target_os = "vxworks", target_os = "rtems")))]
    #[cfg_attr(
        any(
            target_os = "linux",
            target_os = "emscripten",
            target_os = "fuchsia",
            target_os = "l4re",
            target_os = "hurd",
        ),
        link_name = "__errno_location"
    )]
    #[cfg_attr(
        any(
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "cygwin",
            target_os = "android",
            target_os = "redox",
            target_os = "nuttx",
            target_env = "newlib"
        ),
        link_name = "__errno"
    )]
    #[cfg_attr(any(target_os = "solaris", target_os = "illumos"), link_name = "___errno")]
    #[cfg_attr(target_os = "nto", link_name = "__get_errno_ptr")]
    #[cfg_attr(any(target_os = "freebsd", target_vendor = "apple"), link_name = "__error")]
    #[cfg_attr(target_os = "haiku", link_name = "_errnop")]
    #[cfg_attr(target_os = "aix", link_name = "_Errno")]
    // SAFETY: this will always return the same pointer on a given thread.
    #[unsafe(ffi_const)]
    fn errno_location() -> *mut c_int;
}

/// Returns the platform-specific value of errno
#[cfg(not(any(target_os = "dragonfly", target_os = "vxworks", target_os = "rtems")))]
#[inline]
pub fn errno() -> i32 {
    unsafe { (*errno_location()) as i32 }
}

/// Sets the platform-specific value of errno
// needed for readdir and syscall!
#[cfg(all(not(target_os = "dragonfly"), not(target_os = "vxworks"), not(target_os = "rtems")))]
#[allow(dead_code)] // but not all target cfgs actually end up using it
#[inline]
pub fn set_errno(e: i32) {
    unsafe { *errno_location() = e as c_int }
}

#[cfg(target_os = "vxworks")]
#[inline]
pub fn errno() -> i32 {
    unsafe { libc::errnoGet() }
}

#[cfg(target_os = "rtems")]
#[inline]
pub fn errno() -> i32 {
    unsafe extern "C" {
        #[thread_local]
        static _tls_errno: c_int;
    }

    unsafe { _tls_errno as i32 }
}

#[cfg(target_os = "dragonfly")]
#[inline]
pub fn errno() -> i32 {
    unsafe extern "C" {
        #[thread_local]
        static errno: c_int;
    }

    unsafe { errno as i32 }
}

#[cfg(target_os = "dragonfly")]
#[allow(dead_code)]
#[inline]
pub fn set_errno(e: i32) {
    unsafe extern "C" {
        #[thread_local]
        static mut errno: c_int;
    }

    unsafe {
        errno = e;
    }
}

/// Gets a detailed string description for the given error number.
pub fn error_string(errno: i32) -> String {
    unsafe extern "C" {
        #[cfg_attr(
            all(
                any(
                    target_os = "linux",
                    target_os = "hurd",
                    target_env = "newlib",
                    target_os = "cygwin"
                ),
                not(target_env = "ohos")
            ),
            link_name = "__xpg_strerror_r"
        )]
        fn strerror_r(errnum: c_int, buf: *mut c_char, buflen: libc::size_t) -> c_int;
    }

    let mut buf = [0 as c_char; TMPBUF_SZ];

    let p = buf.as_mut_ptr();
    unsafe {
        if strerror_r(errno as c_int, p, buf.len()) < 0 {
            panic!("strerror_r failure");
        }

        let p = p as *const _;
        // We can't always expect a UTF-8 environment. When we don't get that luxury,
        // it's better to give a low-quality error message than none at all.
        String::from_utf8_lossy(CStr::from_ptr(p).to_bytes()).into()
    }
}

#[cfg(target_os = "espidf")]
pub fn getcwd() -> io::Result<PathBuf> {
    Ok(PathBuf::from("/"))
}

#[cfg(not(target_os = "espidf"))]
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

#[cfg(target_os = "espidf")]
pub fn chdir(_p: &path::Path) -> io::Result<()> {
    super::unsupported::unsupported()
}

#[cfg(not(target_os = "espidf"))]
pub fn chdir(p: &path::Path) -> io::Result<()> {
    let result = run_path_with_cstr(p, &|p| unsafe { Ok(libc::chdir(p.as_ptr())) })?;
    if result == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
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

#[cfg(target_os = "aix")]
pub fn current_exe() -> io::Result<PathBuf> {
    #[cfg(test)]
    use realstd::env;

    #[cfg(not(test))]
    use crate::env;
    use crate::io::ErrorKind;

    let exe_path = env::args().next().ok_or(io::const_error!(
        ErrorKind::NotFound,
        "an executable path was not found because no arguments were provided through argv",
    ))?;
    let path = PathBuf::from(exe_path);
    if path.is_absolute() {
        return path.canonicalize();
    }
    // Search PWD to infer current_exe.
    if let Some(pstr) = path.to_str()
        && pstr.contains("/")
    {
        return getcwd().map(|cwd| cwd.join(path))?.canonicalize();
    }
    // Search PATH to infer current_exe.
    if let Some(p) = getenv(OsStr::from_bytes("PATH".as_bytes())) {
        for search_path in split_paths(&p) {
            let pb = search_path.join(&path);
            if pb.is_file()
                && let Ok(metadata) = crate::fs::metadata(&pb)
                && metadata.permissions().mode() & 0o111 != 0
            {
                return pb.canonicalize();
            }
        }
    }
    Err(io::const_error!(ErrorKind::NotFound, "an executable path was not found"))
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
                return Err(io::const_error!(
                    io::ErrorKind::Uncategorized,
                    "KERN_PROC_PATHNAME sysctl returned zero-length string",
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
        Err(io::const_error!(
            io::ErrorKind::Uncategorized,
            "/proc/curproc/exe doesn't point to regular file.",
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
            return Err(io::const_error!(io::ErrorKind::Uncategorized, "no current exe available"));
        }
        let argv0 = CStr::from_ptr(argv[0]).to_bytes();
        if argv0[0] == b'.' || argv0.iter().any(|b| *b == b'/') {
            crate::fs::canonicalize(OsStr::from_bytes(argv0))
        } else {
            Ok(PathBuf::from(OsStr::from_bytes(argv0)))
        }
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "cygwin",
    target_os = "hurd",
    target_os = "android",
    target_os = "nuttx",
    target_os = "emscripten"
))]
pub fn current_exe() -> io::Result<PathBuf> {
    match crate::fs::read_link("/proc/self/exe") {
        Err(ref e) if e.kind() == io::ErrorKind::NotFound => Err(io::const_error!(
            io::ErrorKind::Uncategorized,
            "no /proc/self/exe available. Is /proc mounted?",
        )),
        other => other,
    }
}

#[cfg(target_os = "nto")]
pub fn current_exe() -> io::Result<PathBuf> {
    let mut e = crate::fs::read("/proc/self/exefile")?;
    // Current versions of QNX Neutrino provide a null-terminated path.
    // Ensure the trailing null byte is not returned here.
    if let Some(0) = e.last() {
        e.pop();
    }
    Ok(PathBuf::from(OsString::from_vec(e)))
}

#[cfg(target_vendor = "apple")]
pub fn current_exe() -> io::Result<PathBuf> {
    unsafe {
        let mut sz: u32 = 0;
        #[expect(deprecated)]
        libc::_NSGetExecutablePath(ptr::null_mut(), &mut sz);
        if sz == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut v: Vec<u8> = Vec::with_capacity(sz as usize);
        #[expect(deprecated)]
        let err = libc::_NSGetExecutablePath(v.as_mut_ptr() as *mut i8, &mut sz);
        if err != 0 {
            return Err(io::Error::last_os_error());
        }
        v.set_len(sz as usize - 1); // chop off trailing NUL
        Ok(PathBuf::from(OsString::from_vec(v)))
    }
}

#[cfg(any(target_os = "solaris", target_os = "illumos"))]
pub fn current_exe() -> io::Result<PathBuf> {
    if let Ok(path) = crate::fs::read_link("/proc/self/path/a.out") {
        Ok(path)
    } else {
        unsafe {
            let path = libc::getexecname();
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
}

#[cfg(target_os = "haiku")]
pub fn current_exe() -> io::Result<PathBuf> {
    let mut name = vec![0; libc::PATH_MAX as usize];
    unsafe {
        let result = libc::find_path(
            crate::ptr::null_mut(),
            libc::path_base_directory::B_FIND_PATH_IMAGE_PATH,
            crate::ptr::null_mut(),
            name.as_mut_ptr(),
            name.len(),
        );
        if result != libc::B_OK {
            use crate::io::ErrorKind;
            Err(io::const_error!(ErrorKind::Uncategorized, "error getting executable path"))
        } else {
            // find_path adds the null terminator.
            let name = CStr::from_ptr(name.as_ptr()).to_bytes();
            Ok(PathBuf::from(OsStr::from_bytes(name)))
        }
    }
}

#[cfg(target_os = "redox")]
pub fn current_exe() -> io::Result<PathBuf> {
    crate::fs::read_to_string("/scheme/sys/exe").map(PathBuf::from)
}

#[cfg(target_os = "rtems")]
pub fn current_exe() -> io::Result<PathBuf> {
    crate::fs::read_to_string("sys:exe").map(PathBuf::from)
}

#[cfg(target_os = "l4re")]
pub fn current_exe() -> io::Result<PathBuf> {
    use crate::io::ErrorKind;
    Err(io::const_error!(ErrorKind::Unsupported, "not yet implemented!"))
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

#[cfg(any(target_os = "espidf", target_os = "horizon", target_os = "vita"))]
pub fn current_exe() -> io::Result<PathBuf> {
    super::unsupported::unsupported()
}

#[cfg(target_os = "fuchsia")]
pub fn current_exe() -> io::Result<PathBuf> {
    #[cfg(test)]
    use realstd::env;

    #[cfg(not(test))]
    use crate::env;
    use crate::io::ErrorKind;

    let exe_path = env::args().next().ok_or(io::const_error!(
        ErrorKind::Uncategorized,
        "an executable path was not found because no arguments were provided through argv",
    ))?;
    let path = PathBuf::from(exe_path);

    // Prepend the current working directory to the path if it's not absolute.
    if !path.is_absolute() { getcwd().map(|cwd| cwd.join(path)) } else { Ok(path) }
}

pub struct Env {
    iter: vec::IntoIter<(OsString, OsString)>,
}

// FIXME(https://github.com/rust-lang/rust/issues/114583): Remove this when <OsStr as Debug>::fmt matches <str as Debug>::fmt.
pub struct EnvStrDebug<'a> {
    slice: &'a [(OsString, OsString)],
}

impl fmt::Debug for EnvStrDebug<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { slice } = self;
        f.debug_list()
            .entries(slice.iter().map(|(a, b)| (a.to_str().unwrap(), b.to_str().unwrap())))
            .finish()
    }
}

impl Env {
    pub fn str_debug(&self) -> impl fmt::Debug + '_ {
        let Self { iter } = self;
        EnvStrDebug { slice: iter.as_slice() }
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { iter } = self;
        f.debug_list().entries(iter.as_slice()).finish()
    }
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

// Use `_NSGetEnviron` on Apple platforms.
//
// `_NSGetEnviron` is the documented alternative (see `man environ`), and has
// been available since the first versions of both macOS and iOS.
//
// Nowadays, specifically since macOS 10.8, `environ` has been exposed through
// `libdyld.dylib`, which is linked via. `libSystem.dylib`:
// <https://github.com/apple-oss-distributions/dyld/blob/dyld-1160.6/libdyld/libdyldGlue.cpp#L913>
//
// So in the end, it likely doesn't really matter which option we use, but the
// performance cost of using `_NSGetEnviron` is extremely miniscule, and it
// might be ever so slightly more supported, so let's just use that.
//
// NOTE: The header where this is defined (`crt_externs.h`) was added to the
// iOS 13.0 SDK, which has been the source of a great deal of confusion in the
// past about the availability of this API.
//
// NOTE(madsmtm): Neither this nor using `environ` has been verified to not
// cause App Store rejections; if this is found to be the case, an alternative
// implementation of this is possible using `[NSProcessInfo environment]`
// - which internally uses `_NSGetEnviron` and a system-wide lock on the
// environment variables to protect against `setenv`, so using that might be
// desirable anyhow? Though it also means that we have to link to Foundation.
#[cfg(target_vendor = "apple")]
pub unsafe fn environ() -> *mut *const *const c_char {
    libc::_NSGetEnviron() as *mut *const *const c_char
}

// Use the `environ` static which is part of POSIX.
#[cfg(not(target_vendor = "apple"))]
pub unsafe fn environ() -> *mut *const *const c_char {
    unsafe extern "C" {
        static mut environ: *const *const c_char;
    }
    &raw mut environ
}

static ENV_LOCK: RwLock<()> = RwLock::new(());

pub fn env_read_lock() -> impl Drop {
    ENV_LOCK.read().unwrap_or_else(PoisonError::into_inner)
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
    run_with_cstr(k.as_bytes(), &|k| {
        let _guard = env_read_lock();
        let v = unsafe { libc::getenv(k.as_ptr()) } as *const libc::c_char;

        if v.is_null() {
            Ok(None)
        } else {
            // SAFETY: `v` cannot be mutated while executing this line since we've a read lock
            let bytes = unsafe { CStr::from_ptr(v) }.to_bytes().to_vec();

            Ok(Some(OsStringExt::from_vec(bytes)))
        }
    })
    .ok()
    .flatten()
}

pub unsafe fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    run_with_cstr(k.as_bytes(), &|k| {
        run_with_cstr(v.as_bytes(), &|v| {
            let _guard = ENV_LOCK.write();
            cvt(libc::setenv(k.as_ptr(), v.as_ptr(), 1)).map(drop)
        })
    })
}

pub unsafe fn unsetenv(n: &OsStr) -> io::Result<()> {
    run_with_cstr(n.as_bytes(), &|nbuf| {
        let _guard = ENV_LOCK.write();
        cvt(libc::unsetenv(nbuf.as_ptr())).map(drop)
    })
}

#[cfg(not(target_os = "espidf"))]
pub fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

// Returns the value for [`confstr(key, ...)`][posix_confstr]. Currently only
// used on Darwin, but should work on any unix (in case we need to get
// `_CS_PATH` or `_CS_V[67]_ENV` in the future).
//
// [posix_confstr]:
//     https://pubs.opengroup.org/onlinepubs/9699919799/functions/confstr.html
//
// FIXME: Support `confstr` in Miri.
#[cfg(all(target_vendor = "apple", not(miri)))]
fn confstr(key: c_int, size_hint: Option<usize>) -> io::Result<OsString> {
    let mut buf: Vec<u8> = Vec::with_capacity(0);
    let mut bytes_needed_including_nul = size_hint
        .unwrap_or_else(|| {
            // Treat "None" as "do an extra call to get the length". In theory
            // we could move this into the loop below, but it's hard to do given
            // that it isn't 100% clear if it's legal to pass 0 for `len` when
            // the buffer isn't null.
            unsafe { libc::confstr(key, core::ptr::null_mut(), 0) }
        })
        .max(1);
    // If the value returned by `confstr` is greater than the len passed into
    // it, then the value was truncated, meaning we need to retry. Note that
    // while `confstr` results don't seem to change for a process, it's unclear
    // if this is guaranteed anywhere, so looping does seem required.
    while bytes_needed_including_nul > buf.capacity() {
        // We write into the spare capacity of `buf`. This lets us avoid
        // changing buf's `len`, which both simplifies `reserve` computation,
        // allows working with `Vec<u8>` instead of `Vec<MaybeUninit<u8>>`, and
        // may avoid a copy, since the Vec knows that none of the bytes are needed
        // when reallocating (well, in theory anyway).
        buf.reserve(bytes_needed_including_nul);
        // `confstr` returns
        // - 0 in the case of errors: we break and return an error.
        // - The number of bytes written, iff the provided buffer is enough to
        //   hold the entire value: we break and return the data in `buf`.
        // - Otherwise, the number of bytes needed (including nul): we go
        //   through the loop again.
        bytes_needed_including_nul =
            unsafe { libc::confstr(key, buf.as_mut_ptr().cast::<c_char>(), buf.capacity()) };
    }
    // `confstr` returns 0 in the case of an error.
    if bytes_needed_including_nul == 0 {
        return Err(io::Error::last_os_error());
    }
    // Safety: `confstr(..., buf.as_mut_ptr(), buf.capacity())` returned a
    // non-zero value, meaning `bytes_needed_including_nul` bytes were
    // initialized.
    unsafe {
        buf.set_len(bytes_needed_including_nul);
        // Remove the NUL-terminator.
        let last_byte = buf.pop();
        // ... and smoke-check that it *was* a NUL-terminator.
        assert_eq!(last_byte, Some(0), "`confstr` provided a string which wasn't nul-terminated");
    };
    Ok(OsString::from_vec(buf))
}

#[cfg(all(target_vendor = "apple", not(miri)))]
fn darwin_temp_dir() -> PathBuf {
    confstr(libc::_CS_DARWIN_USER_TEMP_DIR, Some(64)).map(PathBuf::from).unwrap_or_else(|_| {
        // It failed for whatever reason (there are several possible reasons),
        // so return the global one.
        PathBuf::from("/tmp")
    })
}

pub fn temp_dir() -> PathBuf {
    crate::env::var_os("TMPDIR").map(PathBuf::from).unwrap_or_else(|| {
        cfg_if::cfg_if! {
            if #[cfg(all(target_vendor = "apple", not(miri)))] {
                darwin_temp_dir()
            } else if #[cfg(target_os = "android")] {
                PathBuf::from("/data/local/tmp")
            } else {
                PathBuf::from("/tmp")
            }
        }
    })
}

pub fn home_dir() -> Option<PathBuf> {
    return crate::env::var_os("HOME").or_else(|| unsafe { fallback() }).map(PathBuf::from);

    #[cfg(any(
        target_os = "android",
        target_os = "emscripten",
        target_os = "redox",
        target_os = "vxworks",
        target_os = "espidf",
        target_os = "horizon",
        target_os = "vita",
        target_os = "nuttx",
        all(target_vendor = "apple", not(target_os = "macos")),
    ))]
    unsafe fn fallback() -> Option<OsString> {
        None
    }
    #[cfg(not(any(
        target_os = "android",
        target_os = "emscripten",
        target_os = "redox",
        target_os = "vxworks",
        target_os = "espidf",
        target_os = "horizon",
        target_os = "vita",
        target_os = "nuttx",
        all(target_vendor = "apple", not(target_os = "macos")),
    )))]
    unsafe fn fallback() -> Option<OsString> {
        let amt = match libc::sysconf(libc::_SC_GETPW_R_SIZE_MAX) {
            n if n < 0 => 512 as usize,
            n => n as usize,
        };
        let mut buf = Vec::with_capacity(amt);
        let mut p = mem::MaybeUninit::<libc::passwd>::uninit();
        let mut result = ptr::null_mut();
        match libc::getpwuid_r(
            libc::getuid(),
            p.as_mut_ptr(),
            buf.as_mut_ptr(),
            buf.capacity(),
            &mut result,
        ) {
            0 if !result.is_null() => {
                let ptr = (*result).pw_dir as *const _;
                let bytes = CStr::from_ptr(ptr).to_bytes().to_vec();
                Some(OsStringExt::from_vec(bytes))
            }
            _ => None,
        }
    }
}

pub fn exit(code: i32) -> ! {
    crate::sys::exit_guard::unique_thread_exit();
    unsafe { libc::exit(code as c_int) }
}

pub fn getpid() -> u32 {
    unsafe { libc::getpid() as u32 }
}

pub fn getppid() -> u32 {
    unsafe { libc::getppid() as u32 }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub fn glibc_version() -> Option<(usize, usize)> {
    unsafe extern "C" {
        fn gnu_get_libc_version() -> *const libc::c_char;
    }
    let version_cstr = unsafe { CStr::from_ptr(gnu_get_libc_version()) };
    if let Ok(version_str) = version_cstr.to_str() {
        parse_glibc_version(version_str)
    } else {
        None
    }
}

// Returns Some((major, minor)) if the string is a valid "x.y" version,
// ignoring any extra dot-separated parts. Otherwise return None.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn parse_glibc_version(version: &str) -> Option<(usize, usize)> {
    let mut parsed_ints = version.split('.').map(str::parse::<usize>).fuse();
    match (parsed_ints.next(), parsed_ints.next()) {
        (Some(Ok(major)), Some(Ok(minor))) => Some((major, minor)),
        _ => None,
    }
}
