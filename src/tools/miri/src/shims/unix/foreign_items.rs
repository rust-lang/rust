use std::ffi::OsStr;
use std::str;

use rustc_abi::{CanonAbi, Size};
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use self::shims::unix::android::foreign_items as android;
use self::shims::unix::freebsd::foreign_items as freebsd;
use self::shims::unix::linux::foreign_items as linux;
use self::shims::unix::macos::foreign_items as macos;
use self::shims::unix::solarish::foreign_items as solarish;
use crate::concurrency::cpu_affinity::CpuAffinityMask;
use crate::shims::alloc::EvalContextExt as _;
use crate::shims::unix::*;
use crate::{shim_sig, *};

pub fn is_dyn_sym(name: &str, target_os: &str) -> bool {
    match name {
        // Used for tests.
        "isatty" => true,
        // `signal` is set up as a weak symbol in `init_extern_statics` (on Android) so we might as
        // well allow it in `dlsym`.
        "signal" => true,
        // needed at least on macOS to avoid file-based fallback in getrandom
        "getentropy" | "getrandom" => true,
        // Give specific OSes a chance to allow their symbols.
        _ =>
            match target_os {
                "android" => android::is_dyn_sym(name),
                "freebsd" => freebsd::is_dyn_sym(name),
                "linux" => linux::is_dyn_sym(name),
                "macos" => macos::is_dyn_sym(name),
                "solaris" | "illumos" => solarish::is_dyn_sym(name),
                _ => false,
            },
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    // Querying system information
    fn sysconf(&mut self, val: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let name = this.read_scalar(val)?.to_i32()?;
        // FIXME: Which of these are POSIX, and which are GNU/Linux?
        // At least the names seem to all also exist on macOS.
        let sysconfs: &[(&str, fn(&MiriInterpCx<'_>) -> Scalar)] = &[
            ("_SC_PAGESIZE", |this| Scalar::from_int(this.machine.page_size, this.pointer_size())),
            ("_SC_PAGE_SIZE", |this| Scalar::from_int(this.machine.page_size, this.pointer_size())),
            ("_SC_NPROCESSORS_CONF", |this| {
                Scalar::from_int(this.machine.num_cpus, this.pointer_size())
            }),
            ("_SC_NPROCESSORS_ONLN", |this| {
                Scalar::from_int(this.machine.num_cpus, this.pointer_size())
            }),
            // 512 seems to be a reasonable default. The value is not critical, in
            // the sense that getpwuid_r takes and checks the buffer length.
            ("_SC_GETPW_R_SIZE_MAX", |this| Scalar::from_int(512, this.pointer_size())),
            // Miri doesn't have a fixed limit on FDs, but we may be limited in terms of how
            // many *host* FDs we can open. Just use some arbitrary, pretty big value;
            // this can be adjusted if it causes problems.
            // The spec imposes a minimum of `_POSIX_OPEN_MAX` (20).
            ("_SC_OPEN_MAX", |this| Scalar::from_int(2_i32.pow(16), this.pointer_size())),
        ];
        for &(sysconf_name, value) in sysconfs {
            let sysconf_name = this.eval_libc_i32(sysconf_name);
            if sysconf_name == name {
                return interp_ok(value(this));
            }
        }
        throw_unsup_format!("unimplemented sysconf name: {}", name)
    }

    fn strerror_r(
        &mut self,
        errnum: &OpTy<'tcx>,
        buf: &OpTy<'tcx>,
        buflen: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let errnum = this.read_scalar(errnum)?;
        let buf = this.read_pointer(buf)?;
        let buflen = this.read_target_usize(buflen)?;
        let error = this.try_errnum_to_io_error(errnum)?;
        let formatted = match error {
            Some(err) => format!("{err}"),
            None => format!("<unknown errnum in strerror_r: {errnum}>"),
        };
        let (complete, _) = this.write_os_str_to_c_str(OsStr::new(&formatted), buf, buflen)?;
        if complete {
            interp_ok(Scalar::from_i32(0))
        } else {
            interp_ok(Scalar::from_i32(this.eval_libc_i32("ERANGE")))
        }
    }

    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();

        // See `fn emulate_foreign_item_inner` in `shims/foreign_items.rs` for the general pattern.
        match link_name.as_str() {
            // Environment related shims
            "getenv" => {
                let [name] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*const _) -> *mut _),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.getenv(name)?;
                this.write_pointer(result, dest)?;
            }
            "unsetenv" => {
                let [name] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*const _) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.unsetenv(name)?;
                this.write_scalar(result, dest)?;
            }
            "setenv" => {
                let [name, value, overwrite] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*const _, *const _, i32) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                this.read_scalar(overwrite)?.to_i32()?;
                let result = this.setenv(name, value)?;
                this.write_scalar(result, dest)?;
            }
            "getcwd" => {
                let [buf, size] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*mut _, usize) -> *mut _),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.getcwd(buf, size)?;
                this.write_pointer(result, dest)?;
            }
            "chdir" => {
                let [path] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*const _) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.chdir(path)?;
                this.write_scalar(result, dest)?;
            }
            "getpid" => {
                let [] = this.check_shim_sig(
                    shim_sig!(extern "C" fn() -> libc::pid_t),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.getpid()?;
                this.write_scalar(result, dest)?;
            }
            "sysconf" => {
                let [val] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32) -> isize),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.sysconf(val)?;
                this.write_scalar(result, dest)?;
            }
            // File descriptors
            "read" => {
                let [fd, buf, count] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *mut _, usize) -> isize),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let buf = this.read_pointer(buf)?;
                let count = this.read_target_usize(count)?;
                this.read(fd, buf, count, None, dest)?;
            }
            "write" => {
                let [fd, buf, n] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *const _, usize) -> isize),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let buf = this.read_pointer(buf)?;
                let count = this.read_target_usize(n)?;
                trace!("Called write({:?}, {:?}, {:?})", fd, buf, count);
                this.write(fd, buf, count, None, dest)?;
            }
            "pread" => {
                let [fd, buf, count, offset] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *mut _, usize, libc::off_t) -> isize),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let buf = this.read_pointer(buf)?;
                let count = this.read_target_usize(count)?;
                let offset = this.read_scalar(offset)?.to_int(offset.layout.size)?;
                this.read(fd, buf, count, Some(offset), dest)?;
            }
            "pwrite" => {
                let [fd, buf, n, offset] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *const _, usize, libc::off_t) -> isize),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let buf = this.read_pointer(buf)?;
                let count = this.read_target_usize(n)?;
                let offset = this.read_scalar(offset)?.to_int(offset.layout.size)?;
                trace!("Called pwrite({:?}, {:?}, {:?}, {:?})", fd, buf, count, offset);
                this.write(fd, buf, count, Some(offset), dest)?;
            }
            "pread64" => {
                let [fd, buf, count, offset] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *mut _, usize, libc::off64_t) -> isize),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let buf = this.read_pointer(buf)?;
                let count = this.read_target_usize(count)?;
                let offset = this.read_scalar(offset)?.to_int(offset.layout.size)?;
                this.read(fd, buf, count, Some(offset), dest)?;
            }
            "pwrite64" => {
                let [fd, buf, n, offset] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *const _, usize, libc::off64_t) -> isize),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let buf = this.read_pointer(buf)?;
                let count = this.read_target_usize(n)?;
                let offset = this.read_scalar(offset)?.to_int(offset.layout.size)?;
                trace!("Called pwrite64({:?}, {:?}, {:?}, {:?})", fd, buf, count, offset);
                this.write(fd, buf, count, Some(offset), dest)?;
            }
            "close" => {
                let [fd] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.close(fd)?;
                this.write_scalar(result, dest)?;
            }
            "fcntl" => {
                let ([fd_num, cmd], varargs) =
                    this.check_shim_sig_variadic_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.fcntl(fd_num, cmd, varargs)?;
                this.write_scalar(result, dest)?;
            }
            "dup" => {
                let [old_fd] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let old_fd = this.read_scalar(old_fd)?.to_i32()?;
                let new_fd = this.dup(old_fd)?;
                this.write_scalar(new_fd, dest)?;
            }
            "dup2" => {
                let [old_fd, new_fd] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let old_fd = this.read_scalar(old_fd)?.to_i32()?;
                let new_fd = this.read_scalar(new_fd)?.to_i32()?;
                let result = this.dup2(old_fd, new_fd)?;
                this.write_scalar(result, dest)?;
            }
            "flock" => {
                // Currently this function does not exist on all Unixes, e.g. on Solaris.
                this.check_target_os(&["linux", "freebsd", "macos", "illumos"], link_name)?;
                let [fd, op] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let op = this.read_scalar(op)?.to_i32()?;
                let result = this.flock(fd, op)?;
                this.write_scalar(result, dest)?;
            }

            // File and file system access
            "open" | "open64" => {
                // `open` is variadic, the third argument is only present when the second argument
                // has O_CREAT (or on linux O_TMPFILE, but miri doesn't support that) set
                let ([path_raw, flag], varargs) =
                    this.check_shim_sig_variadic_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.open(path_raw, flag, varargs)?;
                this.write_scalar(result, dest)?;
            }
            "unlink" => {
                let [path] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*const _) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.unlink(path)?;
                this.write_scalar(result, dest)?;
            }
            "symlink" => {
                let [target, linkpath] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*const _, *const _) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.symlink(target, linkpath)?;
                this.write_scalar(result, dest)?;
            }
            "rename" => {
                let [oldpath, newpath] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*const _, *const _) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.rename(oldpath, newpath)?;
                this.write_scalar(result, dest)?;
            }
            "mkdir" => {
                let [path, mode] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*const _, libc::mode_t) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.mkdir(path, mode)?;
                this.write_scalar(result, dest)?;
            }
            "rmdir" => {
                let [path] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*const _) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.rmdir(path)?;
                this.write_scalar(result, dest)?;
            }
            "opendir" => {
                let [name] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*const _) -> *mut _),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.opendir(name)?;
                this.write_scalar(result, dest)?;
            }
            "closedir" => {
                let [dirp] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*mut _) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.closedir(dirp)?;
                this.write_scalar(result, dest)?;
            }
            "lseek64" => {
                let [fd, offset, whence] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off64_t, i32) -> libc::off64_t),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let offset = this.read_scalar(offset)?.to_int(offset.layout.size)?;
                let whence = this.read_scalar(whence)?.to_i32()?;
                this.lseek64(fd, offset, whence, dest)?;
            }
            "lseek" => {
                let [fd, offset, whence] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off_t, i32) -> libc::off_t),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let offset = this.read_scalar(offset)?.to_int(offset.layout.size)?;
                let whence = this.read_scalar(whence)?.to_i32()?;
                this.lseek64(fd, offset, whence, dest)?;
            }
            "ftruncate64" => {
                let [fd, length] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off64_t) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let length = this.read_scalar(length)?.to_int(length.layout.size)?;
                let result = this.ftruncate64(fd, length)?;
                this.write_scalar(result, dest)?;
            }
            "ftruncate" => {
                let [fd, length] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off_t) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let fd = this.read_scalar(fd)?.to_i32()?;
                let length = this.read_scalar(length)?.to_int(length.layout.size)?;
                let result = this.ftruncate64(fd, length)?;
                this.write_scalar(result, dest)?;
            }
            "fsync" => {
                let [fd] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.fsync(fd)?;
                this.write_scalar(result, dest)?;
            }
            "fdatasync" => {
                let [fd] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.fdatasync(fd)?;
                this.write_scalar(result, dest)?;
            }
            "readlink" => {
                let [pathname, buf, bufsize] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*const _, *mut _, usize) -> isize),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.readlink(pathname, buf, bufsize)?;
                this.write_scalar(Scalar::from_target_isize(result, this), dest)?;
            }
            "posix_fadvise" => {
                let [fd, offset, len, advice] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off_t, libc::off_t, i32) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                this.read_scalar(fd)?.to_i32()?;
                this.read_scalar(offset)?.to_int(offset.layout.size)?;
                this.read_scalar(len)?.to_int(len.layout.size)?;
                this.read_scalar(advice)?.to_i32()?;
                // fadvise is only informational, we can ignore it.
                this.write_null(dest)?;
            }
            "realpath" => {
                let [path, resolved_path] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*const _, *mut _) -> *mut _),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.realpath(path, resolved_path)?;
                this.write_scalar(result, dest)?;
            }
            "mkstemp" => {
                let [template] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*mut _) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.mkstemp(template)?;
                this.write_scalar(result, dest)?;
            }

            // Unnamed sockets and pipes
            "socketpair" => {
                let [domain, type_, protocol, sv] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32, i32, *mut _) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.socketpair(domain, type_, protocol, sv)?;
                this.write_scalar(result, dest)?;
            }
            "pipe" => {
                let [pipefd] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*mut _) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.pipe2(pipefd, /*flags*/ None)?;
                this.write_scalar(result, dest)?;
            }
            "pipe2" => {
                // Currently this function does not exist on all Unixes, e.g. on macOS.
                this.check_target_os(&["linux", "freebsd", "solaris", "illumos"], link_name)?;
                let [pipefd, flags] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*mut _, i32) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.pipe2(pipefd, Some(flags))?;
                this.write_scalar(result, dest)?;
            }

            // Time
            "gettimeofday" => {
                let [tv, tz] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*mut _, *mut _) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.gettimeofday(tv, tz)?;
                this.write_scalar(result, dest)?;
            }
            "localtime_r" => {
                let [timep, result_op] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(*const _, *mut _) -> *mut _),
                    link_name,
                    abi,
                    args,
                )?;
                let result = this.localtime_r(timep, result_op)?;
                this.write_pointer(result, dest)?;
            }
            "clock_gettime" => {
                let [clk_id, tp] = this.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::clockid_t, *mut _) -> i32),
                    link_name,
                    abi,
                    args,
                )?;
                this.clock_gettime(clk_id, tp, dest)?;
            }

            // Allocation
            "posix_memalign" => {
                let [memptr, align, size] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.posix_memalign(memptr, align, size)?;
                this.write_scalar(result, dest)?;
            }

            "mmap" => {
                let [addr, length, prot, flags, fd, offset] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let offset = this.read_scalar(offset)?.to_int(this.libc_ty_layout("off_t").size)?;
                let ptr = this.mmap(addr, length, prot, flags, fd, offset)?;
                this.write_scalar(ptr, dest)?;
            }
            "munmap" => {
                let [addr, length] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.munmap(addr, length)?;
                this.write_scalar(result, dest)?;
            }

            "reallocarray" => {
                // Currently this function does not exist on all Unixes, e.g. on macOS.
                this.check_target_os(&["linux", "freebsd", "android"], link_name)?;
                let [ptr, nmemb, size] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let nmemb = this.read_target_usize(nmemb)?;
                let size = this.read_target_usize(size)?;
                // reallocarray checks a possible overflow and returns ENOMEM
                // if that happens.
                //
                // Linux: https://www.unix.com/man-page/linux/3/reallocarray/
                // FreeBSD: https://man.freebsd.org/cgi/man.cgi?query=reallocarray
                match this.compute_size_in_bytes(Size::from_bytes(size), nmemb) {
                    None => {
                        this.set_last_error(LibcError("ENOMEM"))?;
                        this.write_null(dest)?;
                    }
                    Some(len) => {
                        let res = this.realloc(ptr, len.bytes())?;
                        this.write_pointer(res, dest)?;
                    }
                }
            }
            "aligned_alloc" => {
                // This is a C11 function, we assume all Unixes have it.
                // (MSVC explicitly does not support this.)
                let [align, size] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let res = this.aligned_alloc(align, size)?;
                this.write_pointer(res, dest)?;
            }

            // Dynamic symbol loading
            "dlsym" => {
                let [handle, symbol] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.read_target_usize(handle)?;
                let symbol = this.read_pointer(symbol)?;
                let name = this.read_c_str(symbol)?;
                if let Ok(name) = str::from_utf8(name)
                    && is_dyn_sym(name, &this.tcx.sess.target.os)
                {
                    let ptr = this.fn_ptr(FnVal::Other(DynSym::from_str(name)));
                    this.write_pointer(ptr, dest)?;
                } else {
                    this.write_null(dest)?;
                }
            }

            // Thread-local storage
            "pthread_key_create" => {
                let [key, dtor] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let key_place = this.deref_pointer_as(key, this.libc_ty_layout("pthread_key_t"))?;
                let dtor = this.read_pointer(dtor)?;

                // Extract the function type out of the signature (that seems easier than constructing it ourselves).
                let dtor = if !this.ptr_is_null(dtor)? {
                    Some(this.get_ptr_fn(dtor)?.as_instance()?)
                } else {
                    None
                };

                // Figure out how large a pthread TLS key actually is.
                // To this end, deref the argument type. This is `libc::pthread_key_t`.
                let key_type = key.layout.ty
                    .builtin_deref(true)
                    .ok_or_else(|| err_ub_format!(
                        "wrong signature used for `pthread_key_create`: first argument must be a raw pointer."
                    ))?;
                let key_layout = this.layout_of(key_type)?;

                // Create key and write it into the memory where `key_ptr` wants it.
                let key = this.machine.tls.create_tls_key(dtor, key_layout.size)?;
                this.write_scalar(Scalar::from_uint(key, key_layout.size), &key_place)?;

                // Return success (`0`).
                this.write_null(dest)?;
            }
            "pthread_key_delete" => {
                let [key] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let key = this.read_scalar(key)?.to_bits(key.layout.size)?;
                this.machine.tls.delete_tls_key(key)?;
                // Return success (0)
                this.write_null(dest)?;
            }
            "pthread_getspecific" => {
                let [key] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let key = this.read_scalar(key)?.to_bits(key.layout.size)?;
                let active_thread = this.active_thread();
                let ptr = this.machine.tls.load_tls(key, active_thread, this)?;
                this.write_scalar(ptr, dest)?;
            }
            "pthread_setspecific" => {
                let [key, new_ptr] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let key = this.read_scalar(key)?.to_bits(key.layout.size)?;
                let active_thread = this.active_thread();
                let new_data = this.read_scalar(new_ptr)?;
                this.machine.tls.store_tls(key, active_thread, new_data, &*this.tcx)?;

                // Return success (`0`).
                this.write_null(dest)?;
            }

            // Synchronization primitives
            "pthread_mutexattr_init" => {
                let [attr] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_mutexattr_init(attr)?;
                this.write_null(dest)?;
            }
            "pthread_mutexattr_settype" => {
                let [attr, kind] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.pthread_mutexattr_settype(attr, kind)?;
                this.write_scalar(result, dest)?;
            }
            "pthread_mutexattr_destroy" => {
                let [attr] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_mutexattr_destroy(attr)?;
                this.write_null(dest)?;
            }
            "pthread_mutex_init" => {
                let [mutex, attr] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_mutex_init(mutex, attr)?;
                this.write_null(dest)?;
            }
            "pthread_mutex_lock" => {
                let [mutex] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_mutex_lock(mutex, dest)?;
            }
            "pthread_mutex_trylock" => {
                let [mutex] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.pthread_mutex_trylock(mutex)?;
                this.write_scalar(result, dest)?;
            }
            "pthread_mutex_unlock" => {
                let [mutex] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.pthread_mutex_unlock(mutex)?;
                this.write_scalar(result, dest)?;
            }
            "pthread_mutex_destroy" => {
                let [mutex] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_mutex_destroy(mutex)?;
                this.write_int(0, dest)?;
            }
            "pthread_rwlock_rdlock" => {
                let [rwlock] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_rwlock_rdlock(rwlock, dest)?;
            }
            "pthread_rwlock_tryrdlock" => {
                let [rwlock] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.pthread_rwlock_tryrdlock(rwlock)?;
                this.write_scalar(result, dest)?;
            }
            "pthread_rwlock_wrlock" => {
                let [rwlock] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_rwlock_wrlock(rwlock, dest)?;
            }
            "pthread_rwlock_trywrlock" => {
                let [rwlock] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.pthread_rwlock_trywrlock(rwlock)?;
                this.write_scalar(result, dest)?;
            }
            "pthread_rwlock_unlock" => {
                let [rwlock] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_rwlock_unlock(rwlock)?;
                this.write_null(dest)?;
            }
            "pthread_rwlock_destroy" => {
                let [rwlock] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_rwlock_destroy(rwlock)?;
                this.write_null(dest)?;
            }
            "pthread_condattr_init" => {
                let [attr] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_condattr_init(attr)?;
                this.write_null(dest)?;
            }
            "pthread_condattr_setclock" => {
                let [attr, clock_id] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.pthread_condattr_setclock(attr, clock_id)?;
                this.write_scalar(result, dest)?;
            }
            "pthread_condattr_getclock" => {
                let [attr, clock_id] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_condattr_getclock(attr, clock_id)?;
                this.write_null(dest)?;
            }
            "pthread_condattr_destroy" => {
                let [attr] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_condattr_destroy(attr)?;
                this.write_null(dest)?;
            }
            "pthread_cond_init" => {
                let [cond, attr] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_cond_init(cond, attr)?;
                this.write_null(dest)?;
            }
            "pthread_cond_signal" => {
                let [cond] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_cond_signal(cond)?;
                this.write_null(dest)?;
            }
            "pthread_cond_broadcast" => {
                let [cond] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_cond_broadcast(cond)?;
                this.write_null(dest)?;
            }
            "pthread_cond_wait" => {
                let [cond, mutex] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_cond_wait(cond, mutex, dest)?;
            }
            "pthread_cond_timedwait" => {
                let [cond, mutex, abstime] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_cond_timedwait(cond, mutex, abstime, dest)?;
            }
            "pthread_cond_destroy" => {
                let [cond] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_cond_destroy(cond)?;
                this.write_null(dest)?;
            }

            // Threading
            "pthread_create" => {
                let [thread, attr, start, arg] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_create(thread, attr, start, arg)?;
                this.write_null(dest)?;
            }
            "pthread_join" => {
                let [thread, retval] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.pthread_join(thread, retval, dest)?;
            }
            "pthread_detach" => {
                let [thread] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let res = this.pthread_detach(thread)?;
                this.write_scalar(res, dest)?;
            }
            "pthread_self" => {
                let [] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let res = this.pthread_self()?;
                this.write_scalar(res, dest)?;
            }
            "sched_yield" => {
                let [] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.sched_yield()?;
                this.write_null(dest)?;
            }
            "nanosleep" => {
                let [duration, rem] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.nanosleep(duration, rem)?;
                this.write_scalar(result, dest)?;
            }
            "clock_nanosleep" => {
                // Currently this function does not exist on all Unixes, e.g. on macOS.
                this.check_target_os(
                    &["freebsd", "linux", "android", "solaris", "illumos"],
                    link_name,
                )?;
                let [clock_id, flags, req, rem] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.clock_nanosleep(clock_id, flags, req, rem)?;
                this.write_scalar(result, dest)?;
            }
            "sched_getaffinity" => {
                // Currently this function does not exist on all Unixes, e.g. on macOS.
                this.check_target_os(&["linux", "freebsd", "android"], link_name)?;
                let [pid, cpusetsize, mask] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let pid = this.read_scalar(pid)?.to_u32()?;
                let cpusetsize = this.read_target_usize(cpusetsize)?;
                let mask = this.read_pointer(mask)?;

                // TODO: when https://github.com/rust-lang/miri/issues/3730 is fixed this should use its notion of tid/pid
                let thread_id = match pid {
                    0 => this.active_thread(),
                    _ =>
                        throw_unsup_format!(
                            "`sched_getaffinity` is only supported with a pid of 0 (indicating the current thread)"
                        ),
                };

                // The mask is stored in chunks, and the size must be a whole number of chunks.
                let chunk_size = CpuAffinityMask::chunk_size(this);

                if this.ptr_is_null(mask)? {
                    this.set_last_error_and_return(LibcError("EFAULT"), dest)?;
                } else if cpusetsize == 0 || cpusetsize.checked_rem(chunk_size).unwrap() != 0 {
                    // we only copy whole chunks of size_of::<c_ulong>()
                    this.set_last_error_and_return(LibcError("EINVAL"), dest)?;
                } else if let Some(cpuset) = this.machine.thread_cpu_affinity.get(&thread_id) {
                    let cpuset = cpuset.clone();
                    // we only copy whole chunks of size_of::<c_ulong>()
                    let byte_count =
                        Ord::min(cpuset.as_slice().len(), cpusetsize.try_into().unwrap());
                    this.write_bytes_ptr(mask, cpuset.as_slice()[..byte_count].iter().copied())?;
                    this.write_null(dest)?;
                } else {
                    // The thread whose ID is pid could not be found
                    this.set_last_error_and_return(LibcError("ESRCH"), dest)?;
                }
            }
            "sched_setaffinity" => {
                // Currently this function does not exist on all Unixes, e.g. on macOS.
                this.check_target_os(&["linux", "freebsd", "android"], link_name)?;
                let [pid, cpusetsize, mask] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let pid = this.read_scalar(pid)?.to_u32()?;
                let cpusetsize = this.read_target_usize(cpusetsize)?;
                let mask = this.read_pointer(mask)?;

                // TODO: when https://github.com/rust-lang/miri/issues/3730 is fixed this should use its notion of tid/pid
                let thread_id = match pid {
                    0 => this.active_thread(),
                    _ =>
                        throw_unsup_format!(
                            "`sched_setaffinity` is only supported with a pid of 0 (indicating the current thread)"
                        ),
                };

                if this.ptr_is_null(mask)? {
                    this.set_last_error_and_return(LibcError("EFAULT"), dest)?;
                } else {
                    // NOTE: cpusetsize might be smaller than `CpuAffinityMask::CPU_MASK_BYTES`.
                    // Any unspecified bytes are treated as zero here (none of the CPUs are configured).
                    // This is not exactly documented, so we assume that this is the behavior in practice.
                    let bits_slice =
                        this.read_bytes_ptr_strip_provenance(mask, Size::from_bytes(cpusetsize))?;
                    // This ignores the bytes beyond `CpuAffinityMask::CPU_MASK_BYTES`
                    let bits_array: [u8; CpuAffinityMask::CPU_MASK_BYTES] =
                        std::array::from_fn(|i| bits_slice.get(i).copied().unwrap_or(0));
                    match CpuAffinityMask::from_array(this, this.machine.num_cpus, bits_array) {
                        Some(cpuset) => {
                            this.machine.thread_cpu_affinity.insert(thread_id, cpuset);
                            this.write_null(dest)?;
                        }
                        None => {
                            // The intersection between the mask and the available CPUs was empty.
                            this.set_last_error_and_return(LibcError("EINVAL"), dest)?;
                        }
                    }
                }
            }

            // Miscellaneous
            "isatty" => {
                let [fd] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.isatty(fd)?;
                this.write_scalar(result, dest)?;
            }
            "pthread_atfork" => {
                let [prepare, parent, child] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.read_pointer(prepare)?;
                this.read_pointer(parent)?;
                this.read_pointer(child)?;
                // We do not support forking, so there is nothing to do here.
                this.write_null(dest)?;
            }
            "getentropy" => {
                // This function is non-standard but exists with the same signature and behavior on
                // Linux, macOS, FreeBSD and Solaris/Illumos.
                this.check_target_os(
                    &["linux", "macos", "freebsd", "illumos", "solaris", "android"],
                    link_name,
                )?;
                let [buf, bufsize] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let buf = this.read_pointer(buf)?;
                let bufsize = this.read_target_usize(bufsize)?;

                // getentropy sets errno to EIO when the buffer size exceeds 256 bytes.
                // FreeBSD: https://man.freebsd.org/cgi/man.cgi?query=getentropy&sektion=3&format=html
                // Linux: https://man7.org/linux/man-pages/man3/getentropy.3.html
                // macOS: https://keith.github.io/xcode-man-pages/getentropy.2.html
                // Solaris/Illumos: https://illumos.org/man/3C/getentropy
                if bufsize > 256 {
                    this.set_last_error_and_return(LibcError("EIO"), dest)?;
                } else {
                    this.gen_random(buf, bufsize)?;
                    this.write_null(dest)?;
                }
            }

            "strerror_r" => {
                let [errnum, buf, buflen] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let result = this.strerror_r(errnum, buf, buflen)?;
                this.write_scalar(result, dest)?;
            }

            "getrandom" => {
                // This function is non-standard but exists with the same signature and behavior on
                // Linux, FreeBSD and Solaris/Illumos.
                this.check_target_os(
                    &["linux", "freebsd", "illumos", "solaris", "android"],
                    link_name,
                )?;
                let [ptr, len, flags] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let len = this.read_target_usize(len)?;
                let _flags = this.read_scalar(flags)?.to_i32()?;
                // We ignore the flags, just always use the same PRNG / host RNG.
                this.gen_random(ptr, len)?;
                this.write_scalar(Scalar::from_target_usize(len, this), dest)?;
            }
            "arc4random_buf" => {
                // This function is non-standard but exists with the same signature and
                // same behavior (eg never fails) on FreeBSD and Solaris/Illumos.
                this.check_target_os(&["freebsd", "illumos", "solaris"], link_name)?;
                let [ptr, len] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let len = this.read_target_usize(len)?;
                this.gen_random(ptr, len)?;
            }
            "_Unwind_RaiseException" => {
                // This is not formally part of POSIX, but it is very wide-spread on POSIX systems.
                // It was originally specified as part of the Itanium C++ ABI:
                // https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html#base-throw.
                // On Linux it is
                // documented as part of the LSB:
                // https://refspecs.linuxfoundation.org/LSB_5.0.0/LSB-Core-generic/LSB-Core-generic/baselib--unwind-raiseexception.html
                // Basically every other UNIX uses the exact same api though. Arm also references
                // back to the Itanium C++ ABI for the definition of `_Unwind_RaiseException` for
                // arm64:
                // https://github.com/ARM-software/abi-aa/blob/main/cppabi64/cppabi64.rst#toc-entry-35
                // For arm32 they did something custom, but similar enough that the same
                // `_Unwind_RaiseException` impl in miri should work:
                // https://github.com/ARM-software/abi-aa/blob/main/ehabi32/ehabi32.rst
                this.check_target_os(
                    &["linux", "freebsd", "illumos", "solaris", "android", "macos"],
                    link_name,
                )?;
                // This function looks and behaves excatly like miri_start_unwind.
                let [payload] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.handle_miri_start_unwind(payload)?;
                return interp_ok(EmulateItemResult::NeedsUnwind);
            }
            "getuid" | "geteuid" => {
                let [] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                // For now, just pretend we always have this fixed UID.
                this.write_int(UID, dest)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "pthread_attr_getguardsize" if this.frame_in_std() => {
                let [_attr, guard_size] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let guard_size_layout = this.machine.layouts.usize;
                let guard_size = this.deref_pointer_as(guard_size, guard_size_layout)?;
                this.write_scalar(
                    Scalar::from_uint(this.machine.page_size, guard_size_layout.size),
                    &guard_size,
                )?;

                // Return success (`0`).
                this.write_null(dest)?;
            }

            "pthread_attr_init" | "pthread_attr_destroy" if this.frame_in_std() => {
                let [_] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.write_null(dest)?;
            }
            "pthread_attr_setstacksize" if this.frame_in_std() => {
                let [_, _] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.write_null(dest)?;
            }

            "pthread_attr_getstack" if this.frame_in_std() => {
                // We don't support "pthread_attr_setstack", so we just pretend all stacks have the same values here.
                // Hence we can mostly ignore the input `attr_place`.
                let [attr_place, addr_place, size_place] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                let _attr_place =
                    this.deref_pointer_as(attr_place, this.libc_ty_layout("pthread_attr_t"))?;
                let addr_place = this.deref_pointer_as(addr_place, this.machine.layouts.usize)?;
                let size_place = this.deref_pointer_as(size_place, this.machine.layouts.usize)?;

                this.write_scalar(
                    Scalar::from_uint(this.machine.stack_addr, this.pointer_size()),
                    &addr_place,
                )?;
                this.write_scalar(
                    Scalar::from_uint(this.machine.stack_size, this.pointer_size()),
                    &size_place,
                )?;

                // Return success (`0`).
                this.write_null(dest)?;
            }

            "signal" | "sigaltstack" if this.frame_in_std() => {
                let [_, _] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.write_null(dest)?;
            }
            "sigaction" | "mprotect" if this.frame_in_std() => {
                let [_, _, _] = this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.write_null(dest)?;
            }

            "getpwuid_r" | "__posix_getpwuid_r" if this.frame_in_std() => {
                // getpwuid_r is the standard name, __posix_getpwuid_r is used on solarish
                let [uid, pwd, buf, buflen, result] =
                    this.check_shim_sig_lenient(abi, CanonAbi::C, link_name, args)?;
                this.check_no_isolation("`getpwuid_r`")?;

                let uid = this.read_scalar(uid)?.to_u32()?;
                let pwd = this.deref_pointer_as(pwd, this.libc_ty_layout("passwd"))?;
                let buf = this.read_pointer(buf)?;
                let buflen = this.read_target_usize(buflen)?;
                let result = this.deref_pointer_as(result, this.machine.layouts.mut_raw_ptr)?;

                // Must be for "us".
                if uid != UID {
                    throw_unsup_format!("`getpwuid_r` on other users is not supported");
                }

                // Reset all fields to `uninit` to make sure nobody reads them.
                // (This is a std-only shim so we are okay with such hacks.)
                this.write_uninit(&pwd)?;

                // We only set the home_dir field.
                #[allow(deprecated)]
                let home_dir = std::env::home_dir().unwrap();
                let (written, _) = this.write_path_to_c_str(&home_dir, buf, buflen)?;
                let pw_dir = this.project_field_named(&pwd, "pw_dir")?;
                this.write_pointer(buf, &pw_dir)?;

                if written {
                    this.write_pointer(pwd.ptr(), &result)?;
                    this.write_null(dest)?;
                } else {
                    this.write_null(&result)?;
                    this.write_scalar(this.eval_libc("ERANGE"), dest)?;
                }
            }

            // Platform-specific shims
            _ => {
                let target_os = &*this.tcx.sess.target.os;
                return match target_os {
                    "android" =>
                        android::EvalContextExt::emulate_foreign_item_inner(
                            this, link_name, abi, args, dest,
                        ),
                    "freebsd" =>
                        freebsd::EvalContextExt::emulate_foreign_item_inner(
                            this, link_name, abi, args, dest,
                        ),
                    "linux" =>
                        linux::EvalContextExt::emulate_foreign_item_inner(
                            this, link_name, abi, args, dest,
                        ),
                    "macos" =>
                        macos::EvalContextExt::emulate_foreign_item_inner(
                            this, link_name, abi, args, dest,
                        ),
                    "solaris" | "illumos" =>
                        solarish::EvalContextExt::emulate_foreign_item_inner(
                            this, link_name, abi, args, dest,
                        ),
                    _ => interp_ok(EmulateItemResult::NotSupported),
                };
            }
        };

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
