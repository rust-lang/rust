//! Implementation of `std::os` functionality for unix systems

#![allow(unused_imports)] // lots of cfg code here

use libc::{c_char, c_int, c_void};

use crate::ffi::{CStr, OsStr, OsString};
use crate::os::unix::prelude::*;
use crate::path::{self, PathBuf};
use crate::sys::cvt;
use crate::sys::helpers::run_path_with_cstr;
use crate::{fmt, io, iter, mem, ptr, slice, str};

const PATH_SEPARATOR: u8 = b':';

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

// This can't just be `impl Iterator` because that requires `'a` to be live on
// drop (see #146045).
pub type SplitPaths<'a> = iter::Map<
    slice::Split<'a, u8, impl FnMut(&u8) -> bool + 'static>,
    impl FnMut(&[u8]) -> PathBuf + 'static,
>;

#[define_opaque(SplitPaths)]
pub fn split_paths(unparsed: &OsStr) -> SplitPaths<'_> {
    fn is_separator(&b: &u8) -> bool {
        b == PATH_SEPARATOR
    }

    fn into_pathbuf(part: &[u8]) -> PathBuf {
        PathBuf::from(OsStr::from_bytes(part))
    }

    unparsed.as_bytes().split(is_separator).map(into_pathbuf)
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

impl crate::error::Error for JoinPathsError {}

#[cfg(target_os = "aix")]
pub fn current_exe() -> io::Result<PathBuf> {
    #[cfg(test)]
    use realstd::env;

    #[cfg(not(test))]
    use crate::env;
    use crate::io;

    let exe_path = env::args().next().ok_or(io::const_error!(
        io::ErrorKind::NotFound,
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
    if let Some(p) = env::var_os(OsStr::from_bytes("PATH".as_bytes())) {
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
    Err(io::const_error!(io::ErrorKind::NotFound, "an executable path was not found"))
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
        Ok(path) => Ok(path),
        Err(ref e) if e.kind() == io::ErrorKind::NotFound => {
            // /proc is not available (e.g., masked in containers, chroot, or unmounted).
            // Fall back to parsing argv[0] and searching PATH.
            #[cfg(test)]
            use realstd::env;

            #[cfg(not(test))]
            use crate::env;

            let exe_path = env::args_os().next().ok_or_else(|| {
                io::const_error!(
                    io::ErrorKind::Uncategorized,
                    "no /proc/self/exe available and no argv[0] provided",
                )
            })?;

            // In test mode, convert from realstd::OsString to local PathBuf via bytes.
            let path = PathBuf::from(OsStr::from_bytes(exe_path.as_encoded_bytes()));

            // If argv[0] is an absolute path, canonicalize it.
            if path.is_absolute() {
                return path.canonicalize();
            }

            // If argv[0] contains a path separator, it's a relative path.
            // Join it with the current working directory and canonicalize.
            if path.as_os_str().as_bytes().contains(&b'/') {
                return getcwd().map(|cwd| cwd.join(path))?.canonicalize();
            }

            // argv[0] is just a command name. Search PATH to find the executable.
            if let Some(path_var) = env::var_os("PATH") {
                // In test mode, convert from realstd::OsString to local OsStr via bytes.
                let path_bytes = path_var.as_encoded_bytes();
                let path_osstr = OsStr::from_bytes(path_bytes);
                for search_path in split_paths(path_osstr) {
                    let candidate = search_path.join(&path);
                    if candidate.is_file() {
                        if let Ok(metadata) = crate::fs::metadata(&candidate) {
                            if metadata.permissions().mode() & 0o111 != 0 {
                                // Canonicalize to resolve symlinks, ensure absolute path,
                                // and clean up path components (matching AIX behavior).
                                return candidate.canonicalize();
                            }
                        }
                    }
                }
            }

            // Could not find the executable in PATH.
            Err(io::const_error!(
                io::ErrorKind::NotFound,
                "no /proc/self/exe available and could not find executable in PATH",
            ))
        }
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
            libc::B_FIND_PATH_IMAGE_PATH,
            crate::ptr::null_mut(),
            name.as_mut_ptr(),
            name.len(),
        );
        if result != libc::B_OK {
            Err(io::const_error!(io::ErrorKind::Uncategorized, "error getting executable path"))
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
    Err(io::const_error!(io::ErrorKind::Unsupported, "not yet implemented!"))
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

    let exe_path = env::args().next().ok_or(io::const_error!(
        io::ErrorKind::Uncategorized,
        "an executable path was not found because no arguments were provided through argv",
    ))?;
    let path = PathBuf::from(exe_path);

    // Prepend the current working directory to the path if it's not absolute.
    if !path.is_absolute() { getcwd().map(|cwd| cwd.join(path)) } else { Ok(path) }
}

#[cfg(all(target_vendor = "apple", not(miri)))]
fn darwin_temp_dir() -> PathBuf {
    crate::sys::pal::conf::confstr(libc::_CS_DARWIN_USER_TEMP_DIR, Some(64))
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // It failed for whatever reason (there are several possible reasons),
            // so return the global one.
            PathBuf::from("/tmp")
        })
}

pub fn temp_dir() -> PathBuf {
    crate::env::var_os("TMPDIR").map(PathBuf::from).unwrap_or_else(|| {
        cfg_select! {
            all(target_vendor = "apple", not(miri)) => darwin_temp_dir(),
            target_os = "android" => PathBuf::from("/data/local/tmp"),
            _ => PathBuf::from("/tmp"),
        }
    })
}

pub fn home_dir() -> Option<PathBuf> {
    return crate::env::var_os("HOME")
        .filter(|s| !s.is_empty())
        .or_else(|| unsafe { fallback() })
        .map(PathBuf::from);

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
