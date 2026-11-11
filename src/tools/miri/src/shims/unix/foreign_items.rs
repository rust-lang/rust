use std::ffi::OsStr;
use std::str;
use std::time::Duration;

use rustc_abi::Size;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;
use rustc_target::spec::{Env, Os, Target};

use self::shims::unix::android::foreign_items as android;
use self::shims::unix::freebsd::foreign_items as freebsd;
use self::shims::unix::linux::foreign_items as linux;
use self::shims::unix::macos::foreign_items as macos;
use self::shims::unix::netbsd::foreign_items as netbsd;
use self::shims::unix::solarish::foreign_items as solarish;
use crate::shims::alloc::EvalContextExt as _;
use crate::shims::cpu_affinity::EvalContextExt as _;
use crate::shims::files::FD_LIMIT;
use crate::shims::unix::*;
use crate::{shim_sig, *};

pub fn is_dyn_sym(name: &str, target: &Target) -> bool {
    match name {
        // Used for (std and Miri) tests.
        "strlen" => true,
        // `signal` is set up as a weak symbol in `init_extern_statics` (on Android) so we might as
        // well allow it in `dlsym`.
        "signal" => true,
        // needed at least on macOS to avoid file-based fallback in getrandom
        "getentropy" | "getrandom" => true,
        // `futimens` is set up as a weak symbol in `init_extern_statics` (on Android), so we
        // allow it here too (it exists on all our Unix targets).
        "futimens" => true,
        // `preadv`/`pwritev` are set up as weak symbols in `init_extern_statics` (on Android,
        // where std uses them via `weak!` since bionic only gained them in API level 24), so
        // we allow them here too. They do not exist on Solaris.
        "preadv" | "pwritev" if !matches!(target.os, Os::Solaris) => true,
        // Give specific OSes a chance to allow their symbols.
        _ =>
            match target.os {
                Os::Android => android::is_dyn_sym(name),
                Os::FreeBsd => freebsd::is_dyn_sym(name),
                Os::Linux => linux::is_dyn_sym(name, target),
                Os::MacOs => macos::is_dyn_sym(name),
                Os::Solaris | Os::Illumos => solarish::is_dyn_sym(name),
                Os::NetBsd => netbsd::is_dyn_sym(name),
                _ => false,
            },
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    // Querying system information
    fn sysconf(&self, val: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_ref();

        let name = ecx.read_scalar(val)?.to_i32()?;
        // FIXME: Which of these are POSIX, and which are GNU/Linux?
        // At least the names seem to all also exist on macOS.
        static SYSCONFS: &[(&str, fn(&MiriInterpCx<'_>) -> i64)] = &[
            ("_SC_PAGESIZE", |ecx| ecx.machine.page_size.try_into().unwrap()),
            ("_SC_PAGE_SIZE", |ecx| ecx.machine.page_size.try_into().unwrap()),
            ("_SC_NPROCESSORS_CONF", |ecx| ecx.machine.num_cpus.into()),
            ("_SC_NPROCESSORS_ONLN", |ecx| ecx.machine.num_cpus.into()),
            // 512 seems to be a reasonable default. The value is not critical, in
            // the sense that getpwuid_r takes and checks the buffer length.
            ("_SC_GETPW_R_SIZE_MAX", |_ecx| 512),
            // We only hand out FDs in `0..FD_LIMIT`, making `FD_LIMIT` the max
            // number of open FDs.
            ("_SC_OPEN_MAX", |_ecx| FD_LIMIT.into()),
            // Our hard-coded hostname is just 4 bytes so we don't need anything big here.
            ("_SC_HOST_NAME_MAX", |_ecx| 255),
        ];
        for &(sysconf_name, value) in SYSCONFS {
            let sysconf_name = ecx.eval_libc_i32(sysconf_name);
            if sysconf_name == name {
                let value = Scalar::from_target_isize(value(ecx), ecx);
                return interp_ok(value);
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
        let ecx = self.eval_context_mut();

        let errnum = ecx.read_scalar(errnum)?;
        let buf = ecx.read_pointer(buf)?;
        let buflen = ecx.read_target_usize(buflen)?;
        let error = ecx.try_errnum_to_io_error(errnum)?;
        let formatted = match error {
            Some(err) => format!("{err}"),
            None => format!("<unknown errnum in strerror_r: {errnum}>"),
        };
        let (complete, _) = ecx.write_os_str_to_c_str(OsStr::new(&formatted), buf, buflen)?;
        if complete {
            interp_ok(Scalar::from_i32(0))
        } else {
            interp_ok(Scalar::from_i32(ecx.eval_libc_i32("ERANGE")))
        }
    }

    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let ecx = self.eval_context_mut();

        if ecx.machine.communicate() {
            // When isolation is disabled we need to check for new host I/O events before
            // running any shimmed function. This is needed to ensure that the shim we
            // execute has up-to-date information about host readiness (as reflected
            // e.g. by epoll) even if the current thread never yields.

            // Perform a non-blocking poll for newly available I/O events from the OS.
            ecx.poll_and_unblock(Some(Duration::ZERO))?;
        }

        // See `fn emulate_foreign_item_inner` in `shims/foreign_items.rs` for the general pattern.
        match link_name.as_str() {
            // Environment related shims
            "getenv" => {
                let [name] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> *_), (link_name, abi, args))?;
                let result = ecx.getenv(name)?;
                ecx.write_pointer(result, dest)?;
            }
            "unsetenv" => {
                let [name] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> i32), (link_name, abi, args))?;
                let result = ecx.unsetenv(name)?;
                ecx.write_scalar(result, dest)?;
            }
            "setenv" => {
                let [name, value, overwrite] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *_, i32) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.read_scalar(overwrite)?.to_i32()?;
                let result = ecx.setenv(name, value)?;
                ecx.write_scalar(result, dest)?;
            }
            "getcwd" => {
                let [buf, size] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize) -> *_),
                    (link_name, abi, args),
                )?;
                let result = ecx.getcwd(buf, size)?;
                ecx.write_pointer(result, dest)?;
            }
            "gethostname" => {
                let [name, len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.gethostname(name, len)?;
                ecx.write_scalar(result, dest)?;
            }
            "chdir" => {
                let [path] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> i32), (link_name, abi, args))?;
                let result = ecx.chdir(path)?;
                ecx.write_scalar(result, dest)?;
            }
            "getpid" => {
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn() -> libc::pid_t),
                    (link_name, abi, args),
                )?;
                let result = ecx.getpid()?;
                ecx.write_scalar(result, dest)?;
            }
            "uname" => {
                // Not all Unixes have the `uname` symbol, e.g. FreeBSD does not.
                ecx.check_target_os(
                    &[Os::Linux, Os::Android, Os::MacOs, Os::Solaris, Os::Illumos],
                    link_name,
                )?;

                let [uname] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::utsname) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.uname(uname, None)?;
                ecx.write_scalar(result, dest)?;
            }
            "sysconf" => {
                let [val] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32) -> isize),
                    (link_name, abi, args),
                )?;
                let result = ecx.sysconf(val)?;
                ecx.write_scalar(result, dest)?;
            }

            // File descriptors
            "read" => {
                let [fd, buf, count] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, usize) -> isize),
                    (link_name, abi, args),
                )?;
                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let buf = ecx.read_pointer(buf)?;
                let count = ecx.read_target_usize(count)?;
                ecx.read(fd, buf, count, None, dest)?;
            }
            "write" => {
                let [fd, buf, n] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, usize) -> isize),
                    (link_name, abi, args),
                )?;
                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let buf = ecx.read_pointer(buf)?;
                let count = ecx.read_target_usize(n)?;
                trace!("Called write({:?}, {:?}, {:?})", fd, buf, count);
                ecx.write(fd, buf, count, None, dest)?;
            }
            "readv" => {
                let [fd, iov, iovcnt] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::iovec, i32) -> isize),
                    (link_name, abi, args),
                )?;
                ecx.readv(fd, iov, iovcnt, None, dest)?;
            }
            "writev" => {
                let [fd, iov, iovcnt] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::iovec, i32) -> isize),
                    (link_name, abi, args),
                )?;
                ecx.writev(fd, iov, iovcnt, None, dest)?;
            }
            "pread" => {
                let [fd, buf, count, offset] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, usize, libc::off_t) -> isize),
                    (link_name, abi, args),
                )?;
                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let buf = ecx.read_pointer(buf)?;
                let count = ecx.read_target_usize(count)?;
                let offset = ecx.read_scalar(offset)?.to_int(offset.layout.size)?;
                ecx.read(fd, buf, count, Some(offset), dest)?;
            }
            "pwrite" => {
                let [fd, buf, n, offset] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, usize, libc::off_t) -> isize),
                    (link_name, abi, args),
                )?;
                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let buf = ecx.read_pointer(buf)?;
                let count = ecx.read_target_usize(n)?;
                let offset = ecx.read_scalar(offset)?.to_int(offset.layout.size)?;
                trace!("Called pwrite({:?}, {:?}, {:?}, {:?})", fd, buf, count, offset);
                ecx.write(fd, buf, count, Some(offset), dest)?;
            }
            "preadv" => {
                let [fd, iov, iovcnt, offset] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::iovec, i32, libc::off_t) -> isize),
                    (link_name, abi, args),
                )?;
                ecx.readv(fd, iov, iovcnt, Some(offset), dest)?;
            }
            "pwritev" => {
                let [fd, iov, iovcnt, offset] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::iovec, i32, libc::off_t) -> isize),
                    (link_name, abi, args),
                )?;
                ecx.writev(fd, iov, iovcnt, Some(offset), dest)?;
            }

            "close" => {
                let [fd] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(i32) -> i32), (link_name, abi, args))?;
                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let result = ecx.close(fd)?;
                ecx.write_scalar(result, dest)?;
            }
            "fcntl" => {
                let ([fd_num, cmd], varargs) = ecx.check_shim_sig_variadic(
                    shim_sig!(extern "C" fn(i32, i32, ...) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.fcntl(&fd_num, &cmd, varargs)?;
                ecx.write_scalar(result, dest)?;
            }
            "dup" => {
                let [old_fd] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(i32) -> i32), (link_name, abi, args))?;
                let old_fd = ecx.read_scalar(old_fd)?.to_i32()?;
                let new_fd = ecx.dup(old_fd)?;
                ecx.write_scalar(new_fd, dest)?;
            }
            "dup2" => {
                let [old_fd, new_fd] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let old_fd = ecx.read_scalar(old_fd)?.to_i32()?;
                let new_fd = ecx.read_scalar(new_fd)?.to_i32()?;
                let result = ecx.dup2(old_fd, new_fd)?;
                ecx.write_scalar(result, dest)?;
            }
            "flock" => {
                // Currently this function does not exist on all Unixes, e.g. on Solaris.
                ecx.check_target_os(
                    &[Os::Linux, Os::Android, Os::FreeBsd, Os::MacOs, Os::Illumos],
                    link_name,
                )?;

                let [fd, op] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let op = ecx.read_scalar(op)?.to_i32()?;
                let result = ecx.flock(fd, op)?;
                ecx.write_scalar(result, dest)?;
            }
            "ioctl" => {
                // The type of `op` depends on the libc. :(
                // glibc, BSD use `unsigned long`, the rest uses `int`.
                let op_is_ulong = match ecx.tcx.sess.target.os {
                    Os::FreeBsd | Os::NetBsd | Os::MacOs => true,
                    Os::Linux if ecx.tcx.sess.target.env == Env::Gnu => true,
                    _ => false,
                };
                let ([fd, op], varargs) = ecx.check_shim_sig_variadic(
                    if op_is_ulong {
                        shim_sig!(extern "C" fn(i32, usize, ...) -> i32)
                    } else {
                        shim_sig!(extern "C" fn(i32, i32, ...) -> i32)
                    },
                    (link_name, abi, args),
                )?;
                let result = ecx.ioctl(&fd, &op, varargs)?;
                ecx.write_scalar(result, dest)?;
            }

            // File and file system access
            "open" => {
                // `open` is variadic, the third argument is only present when the second argument
                // has O_CREAT (or on linux O_TMPFILE, but miri doesn't support that) set
                let ([path_raw, flag], varargs) = ecx.check_shim_sig_variadic(
                    shim_sig!(extern "C" fn(*_, i32, ...) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.open(None, &path_raw, &flag, varargs)?;
                ecx.write_scalar(result, dest)?;
            }
            "openat" => {
                let ([dirfd, path_raw, flag], varargs) = ecx.check_shim_sig_variadic(
                    shim_sig!(extern "C" fn(i32, *_, i32, ...) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.open(Some(&dirfd), &path_raw, &flag, varargs)?;
                ecx.write_scalar(result, dest)?;
            }
            "unlink" => {
                let [path] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> i32), (link_name, abi, args))?;
                let result = ecx.unlink(path)?;
                ecx.write_scalar(result, dest)?;
            }
            "unlinkat" => {
                let [dirfd, path, flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.unlinkat(dirfd, path, flags)?;
                ecx.write_scalar(result, dest)?;
            }
            "symlink" => {
                let [target, linkpath] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *_) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.symlink(target, linkpath)?;
                ecx.write_scalar(result, dest)?;
            }
            "linkat" => {
                let [oldfd, oldpath, newfd, newpath, flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, i32, *_, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.linkat(oldfd, oldpath, newfd, newpath, flags)?;
                ecx.write_scalar(result, dest)?;
            }
            "fstat" => {
                let [fd, buf] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::stat) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.fstat(fd, buf)?;
                ecx.write_scalar(result, dest)?;
            }
            "lstat" => {
                let [path, buf] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *libc::stat) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.lstat(path, buf)?;
                ecx.write_scalar(result, dest)?;
            }
            "stat" => {
                let [path, buf] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *libc::stat) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.stat(path, buf)?;
                ecx.write_scalar(result, dest)?;
            }
            "fstatat" => {
                let [dirfd, path, buf, flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, *libc::stat, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.fstatat(dirfd, path, buf, flags)?;
                ecx.write_scalar(result, dest)?;
            }
            "chmod" => {
                let [path, mode] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, libc::mode_t) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.chmod(path, mode)?;
                ecx.write_scalar(result, dest)?;
            }
            "fchmod" => {
                let [fd, mode] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::mode_t) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.fchmod(fd, mode)?;
                ecx.write_scalar(result, dest)?;
            }
            "rename" => {
                let [oldpath, newpath] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *_) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.rename(oldpath, newpath)?;
                ecx.write_scalar(result, dest)?;
            }
            "mkdir" => {
                let [path, mode] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, libc::mode_t) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.mkdir(path, mode)?;
                ecx.write_scalar(result, dest)?;
            }
            "rmdir" => {
                let [path] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> i32), (link_name, abi, args))?;
                let result = ecx.rmdir(path)?;
                ecx.write_scalar(result, dest)?;
            }
            "opendir" => {
                let [name] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> *_), (link_name, abi, args))?;
                let result = ecx.opendir(name)?;
                ecx.write_scalar(result, dest)?;
            }
            "closedir" => {
                let [dirp] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> i32), (link_name, abi, args))?;
                let result = ecx.closedir(dirp)?;
                ecx.write_scalar(result, dest)?;
            }
            "readdir" => {
                let [dirp] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> *_), (link_name, abi, args))?;
                ecx.readdir(dirp, dest, "dirent")?;
            }
            "dirfd" => {
                let [dirp] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> i32), (link_name, abi, args))?;
                ecx.dirfd(dirp, dest)?;
            }
            "lseek" => {
                let [fd, offset, whence] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off_t, i32) -> libc::off_t),
                    (link_name, abi, args),
                )?;
                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let offset = ecx.read_scalar(offset)?.to_int(offset.layout.size)?;
                let whence = ecx.read_scalar(whence)?.to_i32()?;
                ecx.lseek(fd, offset, whence, dest)?;
            }
            "ftruncate" => {
                let [fd, length] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off_t) -> i32),
                    (link_name, abi, args),
                )?;
                let fd = ecx.read_scalar(fd)?.to_i32()?;
                let length = ecx.read_scalar(length)?.to_int(length.layout.size)?;
                let result = ecx.ftruncate64(fd, length)?;
                ecx.write_scalar(result, dest)?;
            }
            "fsync" => {
                let [fd] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(i32) -> i32), (link_name, abi, args))?;
                let result = ecx.fsync(fd)?;
                ecx.write_scalar(result, dest)?;
            }
            "fdatasync" => {
                // fdatasync is not supported by macos.
                ecx.check_target_os(
                    &[Os::Linux, Os::FreeBsd, Os::NetBsd, Os::Solaris, Os::Illumos, Os::Android],
                    link_name,
                )?;
                let [fd] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(i32) -> i32), (link_name, abi, args))?;
                let result = ecx.fdatasync(fd)?;
                ecx.write_scalar(result, dest)?;
            }
            "futimens" => {
                let [fd, times] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::timespec) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.futimens(fd, times)?;
                ecx.write_scalar(result, dest)?;
            }
            "utimensat" => {
                let [dirfd, path, times, flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, *libc::timespec, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.utimensat(dirfd, path, times, flags)?;
                ecx.write_scalar(result, dest)?;
            }
            "readlink" => {
                let [pathname, buf, bufsize] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *_, usize) -> isize),
                    (link_name, abi, args),
                )?;
                let result = ecx.readlink(pathname, buf, bufsize)?;
                ecx.write_scalar(Scalar::from_target_isize(result, ecx), dest)?;
            }
            "posix_fadvise" => {
                let [fd, offset, len, advice] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off_t, libc::off_t, i32) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.read_scalar(fd)?.to_i32()?;
                ecx.read_scalar(offset)?.to_int(offset.layout.size)?;
                ecx.read_scalar(len)?.to_int(len.layout.size)?;
                ecx.read_scalar(advice)?.to_i32()?;
                // fadvise is only informational, we can ignore it.
                ecx.write_int(0, dest)?;
            }

            "posix_fallocate" => {
                // posix_fallocate is not supported by macos.
                ecx.check_target_os(
                    &[Os::Linux, Os::FreeBsd, Os::Solaris, Os::Illumos, Os::Android],
                    link_name,
                )?;

                let [fd, offset, len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::off_t, libc::off_t) -> i32),
                    (link_name, abi, args),
                )?;

                let fd = ecx.read_scalar(fd)?.to_i32()?;
                // We don't support platforms which have libc::off_t bigger than 64 bits.
                let offset =
                    i64::try_from(ecx.read_scalar(offset)?.to_int(offset.layout.size)?).unwrap();
                let len = i64::try_from(ecx.read_scalar(len)?.to_int(len.layout.size)?).unwrap();

                let result = ecx.posix_fallocate(fd, offset, len)?;
                ecx.write_scalar(result, dest)?;
            }

            "realpath" => {
                let [path, resolved_path] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *_) -> *_),
                    (link_name, abi, args),
                )?;
                let result = ecx.realpath(path, resolved_path)?;
                ecx.write_scalar(result, dest)?;
            }
            "mkstemp" => {
                let [template] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> i32), (link_name, abi, args))?;
                let result = ecx.mkstemp(template)?;
                ecx.write_scalar(result, dest)?;
            }

            // Poll
            "poll" => {
                let [fds, nfds, timeout] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pollfd, libc::nfds_t, i32) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.poll(fds, nfds, timeout, dest)?;
            }

            // Sockets and pipes
            "socketpair" => {
                let [domain, type_, protocol, sv] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32, i32, *i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.socketpair(domain, type_, protocol, sv)?;
                ecx.write_scalar(result, dest)?;
            }
            "pipe" => {
                let [pipefd] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.pipe2(pipefd, /*flags*/ None)?;
                ecx.write_scalar(result, dest)?;
            }
            "pipe2" => {
                // Currently this function does not exist on all Unixes, e.g. on macOS.
                ecx.check_target_os(
                    &[Os::Linux, Os::Android, Os::FreeBsd, Os::Solaris, Os::Illumos],
                    link_name,
                )?;

                let [pipefd, flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*i32, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.pipe2(pipefd, Some(flags))?;
                ecx.write_scalar(result, dest)?;
            }
            "socket" => {
                let [domain, type_, protocol] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.socket(domain, type_, protocol)?;
                ecx.write_scalar(result, dest)?;
            }
            "bind" => {
                let [socket, address, address_len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::sockaddr, libc::socklen_t) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.bind(socket, address, address_len)?;
                ecx.write_scalar(result, dest)?;
            }
            "listen" => {
                let [socket, backlog] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.listen(socket, backlog)?;
                ecx.write_scalar(result, dest)?;
            }
            "accept" => {
                let [socket, address, address_len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::sockaddr, *libc::socklen_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.accept4(socket, address, address_len, /* flags */ None, dest)?;
            }
            "accept4" => {
                let [socket, address, address_len, flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::sockaddr, *libc::socklen_t, i32) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.accept4(socket, address, address_len, Some(flags), dest)?;
            }
            "connect" => {
                let [socket, address, address_len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::sockaddr, libc::socklen_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.connect(socket, address, address_len, dest)?;
            }
            "send" => {
                let [socket, buffer, length, flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, libc::size_t, i32) -> libc::ssize_t),
                    (link_name, abi, args),
                )?;
                ecx.send(socket, buffer, length, flags, /* dest_addr */ None, dest)?;
            }
            "sendto" => {
                let [socket, buffer, length, flags, dest_addr, dest_addr_len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, libc::size_t, i32, *libc::sockaddr, libc::socklen_t) -> libc::ssize_t),
                    (link_name, abi, args),
                )?;
                ecx.send(socket, buffer, length, flags, Some((dest_addr, dest_addr_len)), dest)?;
            }
            "recv" => {
                let [socket, buffer, length, flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, libc::size_t, i32) -> libc::ssize_t),
                    (link_name, abi, args),
                )?;
                ecx.recv(socket, buffer, length, flags, /* dest_addr */ None, dest)?;
            }
            "recvfrom" => {
                let [socket, buffer, length, flags, dest_addr, dest_addr_len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, libc::size_t, i32, *libc::sockaddr, *libc::socklen_t) -> libc::ssize_t),
                    (link_name, abi, args),
                )?;
                ecx.recv(socket, buffer, length, flags, Some((dest_addr, dest_addr_len)), dest)?;
            }
            "setsockopt" => {
                let [socket, level, option_name, option_value, option_len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32, i32, *_, libc::socklen_t) -> i32),
                    (link_name, abi, args),
                )?;
                let result =
                    ecx.setsockopt(socket, level, option_name, option_value, option_len)?;
                ecx.write_scalar(result, dest)?;
            }
            "getsockopt" => {
                let [socket, level, option_name, option_value, option_len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32, i32, *_, *_) -> i32),
                    (link_name, abi, args),
                )?;
                let result =
                    ecx.getsockopt(socket, level, option_name, option_value, option_len)?;
                ecx.write_scalar(result, dest)?;
            }
            "getsockname" => {
                let [socket, address, address_len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::sockaddr, *libc::socklen_t) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.getsockname(socket, address, address_len)?;
                ecx.write_scalar(result, dest)?;
            }
            "getpeername" => {
                let [socket, address, address_len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *libc::sockaddr, *libc::socklen_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.getpeername(socket, address, address_len, dest)?;
            }
            "shutdown" => {
                let [sockfd, how] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.shutdown(sockfd, how)?;
                ecx.write_scalar(result, dest)?;
            }
            "getaddrinfo" => {
                let [node, service, hints, res] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *_, *_, **_) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.getaddrinfo(node, service, hints, res)?;
                ecx.write_scalar(result, dest)?;
            }
            "freeaddrinfo" => {
                let [res] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> ()), (link_name, abi, args))?;
                ecx.freeaddrinfo(res)?;
            }

            // Time
            "gettimeofday" => {
                let [tv, tz] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::timeval, *libc::timezone) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.gettimeofday(tv, tz)?;
                ecx.write_scalar(result, dest)?;
            }
            "localtime_r" => {
                let [timep, result_op] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::time_t, *libc::tm) -> *libc::tm),
                    (link_name, abi, args),
                )?;
                let result = ecx.localtime_r(timep, result_op)?;
                ecx.write_pointer(result, dest)?;
            }
            "clock_gettime" => {
                let [clk_id, tp] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::clockid_t, *libc::timespec) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.clock_gettime(clk_id, tp, dest)?;
            }

            // Allocation
            "posix_memalign" => {
                let [memptr, align, size] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(**_, usize, usize) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.posix_memalign(memptr, align, size)?;
                ecx.write_scalar(result, dest)?;
            }

            "mmap" => {
                let [addr, length, prot, flags, fd, offset] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize, i32, i32, i32, libc::off_t) -> *_),
                    (link_name, abi, args),
                )?;
                let offset = ecx.read_scalar(offset)?.to_int(ecx.libc_ty_layout("off_t").size)?;
                let ptr = ecx.mmap(addr, length, prot, flags, fd, offset)?;
                ecx.write_scalar(ptr, dest)?;
            }
            "munmap" => {
                let [addr, length] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.munmap(addr, length)?;
                ecx.write_scalar(result, dest)?;
            }
            "mprotect" => {
                let [addr, length, prot] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.mprotect(addr, length, prot)?;
                ecx.write_scalar(result, dest)?;
            }
            "madvise" => {
                let [addr, length, advice] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.madvise(addr, length, advice)?;
                ecx.write_scalar(result, dest)?;
            }

            "reallocarray" => {
                // Currently this function does not exist on all Unixes, e.g. on macOS.
                ecx.check_target_os(&[Os::Linux, Os::FreeBsd, Os::Android], link_name)?;

                let [ptr, nmemb, size] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize, usize) -> *_),
                    (link_name, abi, args),
                )?;
                let ptr = ecx.read_pointer(ptr)?;
                let nmemb = ecx.read_target_usize(nmemb)?;
                let size = ecx.read_target_usize(size)?;
                // reallocarray checks a possible overflow and returns ENOMEM
                // if that happens.
                //
                // Linux: https://www.unix.com/man-page/linux/3/reallocarray/
                // FreeBSD: https://man.freebsd.org/cgi/man.cgi?query=reallocarray
                match ecx.compute_size_in_bytes(Size::from_bytes(size), nmemb) {
                    None => {
                        ecx.set_last_error(LibcError("ENOMEM"))?;
                        ecx.write_int(0, dest)?;
                    }
                    Some(len) => {
                        let res = ecx.realloc(ptr, len.bytes())?;
                        ecx.write_pointer(res, dest)?;
                    }
                }
            }
            "aligned_alloc" => {
                // This is a C11 function, we assume all Unixes have it.
                // (MSVC explicitly does not support this.)
                let [align, size] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(usize, usize) -> *_),
                    (link_name, abi, args),
                )?;
                let res = ecx.aligned_alloc(align, size)?;
                ecx.write_pointer(res, dest)?;
            }

            // Dynamic symbol loading
            "dlsym" => {
                let [handle, symbol] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *_) -> *_),
                    (link_name, abi, args),
                )?;
                ecx.read_target_usize(handle)?;
                let symbol = ecx.read_pointer(symbol)?;
                let name = ecx.read_c_str(symbol)?;
                let Ok(name) = str::from_utf8(name) else {
                    throw_unsup_format!("dlsym: non UTF-8 symbol name not supported")
                };
                if ecx.is_dyn_sym(name) {
                    let ptr = ecx.fn_ptr(FnVal::Other(DynSym::from_str(name)));
                    ecx.write_pointer(ptr, dest)?;
                } else if let Some(&ptr) = ecx.machine.extern_statics.get(&Symbol::intern(name)) {
                    ecx.write_pointer(ptr, dest)?;
                } else {
                    ecx.write_int(0, dest)?;
                }
            }

            // Thread-local storage
            "pthread_key_create" => {
                let [key, dtor] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_key_t, fn(..) -> _) -> i32),
                    (link_name, abi, args),
                )?;
                let key_place = ecx.deref_pointer(key)?;
                let dtor = ecx.read_pointer(dtor)?;

                // Extract the function type out of the signature (that seems easier than constructing it ourselves).
                let dtor = if !ecx.ptr_is_null(dtor)? {
                    Some((
                        ecx.get_ptr_fn(dtor)?.as_instance()?,
                        ecx.machine.current_user_relevant_span(),
                    ))
                } else {
                    None
                };

                // Create key and write it into the memory where `key_ptr` wants it.
                let key = ecx.machine.tls.create_tls_key(dtor, key_place.layout.size)?;
                ecx.write_scalar(Scalar::from_uint(key, key_place.layout.size), &key_place)?;

                // Return success (`0`).
                ecx.write_int(0, dest)?;
            }
            "pthread_key_delete" => {
                let [key] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_key_t) -> i32),
                    (link_name, abi, args),
                )?;
                let key = ecx.read_scalar(key)?.to_bits(key.layout.size)?;
                ecx.machine.tls.delete_tls_key(key)?;
                // Return success (0)
                ecx.write_int(0, dest)?;
            }
            "pthread_getspecific" => {
                let [key] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_key_t) -> *_),
                    (link_name, abi, args),
                )?;
                let key = ecx.read_scalar(key)?.to_bits(key.layout.size)?;
                let active_thread = ecx.active_thread();
                let ptr = ecx.machine.tls.load_tls_key(key, active_thread)?;
                ecx.write_pointer(ptr, dest)?;
            }
            "pthread_setspecific" => {
                let [key, new_ptr] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_key_t, *_) -> i32),
                    (link_name, abi, args),
                )?;
                let key = ecx.read_scalar(key)?.to_bits(key.layout.size)?;
                let active_thread = ecx.active_thread();
                let new_data = ecx.read_pointer(new_ptr)?;
                ecx.machine.tls.store_tls_key(key, active_thread, new_data)?;

                // Return success (`0`).
                ecx.write_int(0, dest)?;
            }

            // Synchronization primitives
            "pthread_mutexattr_init" => {
                let [attr] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_mutexattr_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_mutexattr_init(attr)?;
                ecx.write_int(0, dest)?;
            }
            "pthread_mutexattr_settype" => {
                let [attr, kind] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_mutexattr_t, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.pthread_mutexattr_settype(attr, kind)?;
                ecx.write_scalar(result, dest)?;
            }
            "pthread_mutexattr_destroy" => {
                let [attr] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_mutexattr_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_mutexattr_destroy(attr)?;
                ecx.write_int(0, dest)?;
            }
            "pthread_mutex_init" => {
                let [mutex, attr] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_mutex_t, *libc::pthread_mutexattr_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_mutex_init(mutex, attr)?;
                ecx.write_int(0, dest)?;
            }
            "pthread_mutex_lock" => {
                let [mutex] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_mutex_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_mutex_lock(mutex, dest)?;
            }
            "pthread_mutex_trylock" => {
                let [mutex] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_mutex_t) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.pthread_mutex_trylock(mutex)?;
                ecx.write_scalar(result, dest)?;
            }
            "pthread_mutex_unlock" => {
                let [mutex] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_mutex_t) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.pthread_mutex_unlock(mutex)?;
                ecx.write_scalar(result, dest)?;
            }
            "pthread_mutex_destroy" => {
                let [mutex] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_mutex_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_mutex_destroy(mutex)?;
                ecx.write_int(0, dest)?;
            }
            "pthread_rwlock_rdlock" => {
                let [rwlock] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_rwlock_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_rwlock_rdlock(rwlock, dest)?;
            }
            "pthread_rwlock_tryrdlock" => {
                let [rwlock] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_rwlock_t) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.pthread_rwlock_tryrdlock(rwlock)?;
                ecx.write_scalar(result, dest)?;
            }
            "pthread_rwlock_wrlock" => {
                let [rwlock] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_rwlock_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_rwlock_wrlock(rwlock, dest)?;
            }
            "pthread_rwlock_trywrlock" => {
                let [rwlock] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_rwlock_t) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.pthread_rwlock_trywrlock(rwlock)?;
                ecx.write_scalar(result, dest)?;
            }
            "pthread_rwlock_unlock" => {
                let [rwlock] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_rwlock_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_rwlock_unlock(rwlock)?;
                ecx.write_int(0, dest)?;
            }
            "pthread_rwlock_destroy" => {
                let [rwlock] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_rwlock_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_rwlock_destroy(rwlock)?;
                ecx.write_int(0, dest)?;
            }
            "pthread_condattr_init" => {
                let [attr] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_condattr_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_condattr_init(attr)?;
                ecx.write_int(0, dest)?;
            }
            "pthread_condattr_setclock" => {
                let [attr, clock_id] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_condattr_t, i32) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.pthread_condattr_setclock(attr, clock_id)?;
                ecx.write_scalar(result, dest)?;
            }
            "pthread_condattr_getclock" => {
                let [attr, clock_id] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_condattr_t, *libc::clockid_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_condattr_getclock(attr, clock_id)?;
                ecx.write_int(0, dest)?;
            }
            "pthread_condattr_destroy" => {
                let [attr] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_condattr_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_condattr_destroy(attr)?;
                ecx.write_int(0, dest)?;
            }
            "pthread_cond_init" => {
                let [cond, attr] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_cond_t, *libc::pthread_condattr_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_cond_init(cond, attr)?;
                ecx.write_int(0, dest)?;
            }
            "pthread_cond_signal" => {
                let [cond] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_cond_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_cond_signal(cond)?;
                ecx.write_int(0, dest)?;
            }
            "pthread_cond_broadcast" => {
                let [cond] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_cond_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_cond_broadcast(cond)?;
                ecx.write_int(0, dest)?;
            }
            "pthread_cond_wait" => {
                let [cond, mutex] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_cond_t, *libc::pthread_mutex_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_cond_wait(cond, mutex, dest)?;
            }
            "pthread_cond_timedwait" => {
                let [cond, mutex, abstime] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_cond_t, *libc::pthread_mutex_t, *libc::timespec) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_cond_timedwait(
                    cond, mutex, abstime, dest, /* macos_relative_np */ false,
                )?;
            }
            "pthread_cond_destroy" => {
                let [cond] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_cond_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_cond_destroy(cond)?;
                ecx.write_int(0, dest)?;
            }
            "sem_init" => {
                let [sem, pshared, value] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::sem_t, i32, u32) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_sem_init(sem, pshared, value, dest)?;
            }
            "sem_wait" => {
                let [sem] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::sem_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_sem_wait(sem, dest)?;
            }
            "sem_timedwait" => {
                let [sem, abs_timeout] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::sem_t, *libc::timespec) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_sem_timedwait(sem, abs_timeout, dest)?;
            }
            "sem_post" => {
                let [sem] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::sem_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_sem_post(sem, dest)?;
            }
            "sem_destroy" => {
                let [sem] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::sem_t) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_sem_destroy(sem, dest)?;
            }

            // Threading
            "pthread_create" => {
                let [thread, attr, start, arg] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_t, *_, fn(..) -> _, *_) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_create(thread, attr, start, arg)?;
                ecx.write_int(0, dest)?;
            }
            "pthread_join" => {
                let [thread, retval] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t, *_) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.pthread_join(thread, retval, dest)?;
            }
            "pthread_detach" => {
                let [thread] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pthread_t) -> i32),
                    (link_name, abi, args),
                )?;
                let res = ecx.pthread_detach(thread)?;
                ecx.write_scalar(res, dest)?;
            }
            "pthread_self" => {
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn() -> libc::pthread_t),
                    (link_name, abi, args),
                )?;
                let res = ecx.pthread_self()?;
                ecx.write_scalar(res, dest)?;
            }
            "sched_yield" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> i32), (link_name, abi, args))?;
                ecx.sched_yield()?;
                ecx.write_int(0, dest)?;
            }
            "nanosleep" => {
                let [duration, rem] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::timespec, *_) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.nanosleep(duration, rem)?;
                ecx.write_scalar(result, dest)?;
            }
            "clock_nanosleep" => {
                // Currently this function does not exist on all Unixes, e.g. on macOS.
                ecx.check_target_os(
                    &[Os::FreeBsd, Os::Linux, Os::Android, Os::Solaris, Os::Illumos],
                    link_name,
                )?;

                let [clock_id, flags, req, rem] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::clockid_t, i32, *libc::timespec, *_) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.clock_nanosleep(clock_id, flags, req, rem)?;
                ecx.write_scalar(result, dest)?;
            }
            "sched_getaffinity" => {
                // Currently this function does not exist on all Unixes, e.g. on macOS.
                ecx.check_target_os(&[Os::Linux, Os::FreeBsd, Os::Android], link_name)?;

                let [pid, cpusetsize, mask] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pid_t, usize, *_) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.sched_getaffinity(pid, cpusetsize, mask, dest)?;
            }
            "sched_setaffinity" => {
                // Currently this function does not exist on all Unixes, e.g. on macOS.
                ecx.check_target_os(&[Os::Linux, Os::FreeBsd, Os::Android], link_name)?;

                let [pid, cpusetsize, mask] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::pid_t, usize, *_) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.sched_setaffinity(pid, cpusetsize, mask, dest)?;
            }

            // Miscellaneous
            "isatty" => {
                let [fd] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(i32) -> i32), (link_name, abi, args))?;
                let result = ecx.isatty(fd)?;
                ecx.write_scalar(result, dest)?;
            }
            "pthread_atfork" => {
                let [prepare, parent, child] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(fn(..) -> _, fn(..) -> _, fn(..) -> _) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.read_pointer(prepare)?;
                ecx.read_pointer(parent)?;
                ecx.read_pointer(child)?;
                // We do not support forking, so there is nothing to do here.
                ecx.write_int(0, dest)?;
            }
            "strerror_r" => {
                let [errnum, buf, buflen] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, usize) -> i32),
                    (link_name, abi, args),
                )?;
                let result = ecx.strerror_r(errnum, buf, buflen)?;
                ecx.write_scalar(result, dest)?;
            }
            "getentropy" => {
                // This function is non-standard but exists with the same signature and behavior on
                // Linux, macOS, FreeBSD and Solaris/Illumos.
                ecx.check_target_os(
                    &[Os::Linux, Os::MacOs, Os::FreeBsd, Os::Illumos, Os::Solaris, Os::Android],
                    link_name,
                )?;

                let [buf, bufsize] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize) -> i32),
                    (link_name, abi, args),
                )?;
                let buf = ecx.read_pointer(buf)?;
                let bufsize = ecx.read_target_usize(bufsize)?;

                // getentropy sets errno to EIO when the buffer size exceeds 256 bytes.
                // FreeBSD: https://man.freebsd.org/cgi/man.cgi?query=getentropy&sektion=3&format=html
                // Linux: https://man7.org/linux/man-pages/man3/getentropy.3.html
                // macOS: https://keith.github.io/xcode-man-pages/getentropy.2.html
                // Solaris/Illumos: https://illumos.org/man/3C/getentropy
                if bufsize > 256 {
                    ecx.set_errno_and_return_neg1(LibcError("EIO"), dest)?;
                } else {
                    ecx.gen_random(buf, bufsize)?;
                    ecx.write_int(0, dest)?;
                }
            }
            "getrandom" => {
                // This function is non-standard but exists with the same signature and behavior on
                // Linux, FreeBSD and Solaris/Illumos.
                ecx.check_target_os(
                    &[Os::Linux, Os::FreeBsd, Os::Illumos, Os::Solaris, Os::Android],
                    link_name,
                )?;

                let [ptr, len, flags] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize, u32) -> isize),
                    (link_name, abi, args),
                )?;
                let ptr = ecx.read_pointer(ptr)?;
                let len = ecx.read_target_usize(len)?;
                let _flags = ecx.read_scalar(flags)?.to_i32()?;
                // We ignore the flags, just always use the same PRNG / host RNG.
                ecx.gen_random(ptr, len)?;
                ecx.write_scalar(Scalar::from_target_usize(len, ecx), dest)?;
            }
            "arc4random_buf" => {
                // This function is non-standard but exists with the same signature and
                // same behavior (eg never fails) on FreeBSD and Solaris/Illumos.
                ecx.check_target_os(&[Os::FreeBsd, Os::Illumos, Os::Solaris], link_name)?;

                let [ptr, len] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize) -> ()),
                    (link_name, abi, args),
                )?;
                let ptr = ecx.read_pointer(ptr)?;
                let len = ecx.read_target_usize(len)?;
                ecx.gen_random(ptr, len)?;
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
                ecx.check_target_os(
                    &[Os::Linux, Os::FreeBsd, Os::Illumos, Os::Solaris, Os::Android, Os::MacOs],
                    link_name,
                )?;

                // This function looks and behaves exactly like miri_start_unwind.
                let [payload] = &ecx.check_shim_sig(
                    // Look up the return type via `panic_unwind::`, not via `unwind::`, as
                    // the latter it not always unique.
                    shim_sig!(extern "C" fn(*_) -> panic_unwind::imp::uw::_Unwind_Reason_Code),
                    (link_name, abi, args),
                )?;
                ecx.handle_miri_start_unwind(payload)?;
                return interp_ok(EmulateItemResult::NeedsUnwind);
            }
            "getuid" | "geteuid" => {
                let [] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn() -> libc::uid_t),
                    (link_name, abi, args),
                )?;
                // For now, just pretend we always have this fixed UID.
                ecx.write_int(UID, dest)?;
            }
            "pause" => {
                let [] =
                    &ecx.check_shim_sig(shim_sig!(extern "C" fn() -> i32), (link_name, abi, args))?;
                // This blocks the current thread until a signal gets delivered. We don't support
                // signals, so we block forever.
                ecx.block_thread(
                    BlockReason::Pause,
                    None,
                    callback!(@capture<'tcx> {}
                    |_ecx, _unblock: UnblockKind| {
                        panic!("should never unblock")
                    }),
                );
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "pthread_attr_getguardsize" if ecx.frame_in_std() => {
                let [_attr, guard_size] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *usize) -> i32),
                    (link_name, abi, args),
                )?;
                let guard_size = ecx.deref_pointer(guard_size)?;
                ecx.write_scalar(
                    Scalar::from_uint(ecx.machine.page_size, guard_size.layout.size),
                    &guard_size,
                )?;

                // Return success (`0`).
                ecx.write_int(0, dest)?;
            }

            "pthread_attr_init" | "pthread_attr_destroy" if ecx.frame_in_std() => {
                let [_] = &ecx
                    .check_shim_sig(shim_sig!(extern "C" fn(*_) -> i32), (link_name, abi, args))?;
                ecx.write_int(0, dest)?;
            }

            "pthread_attr_setstacksize" if ecx.frame_in_std() => {
                let [_, _] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, usize) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.write_int(0, dest)?;
            }

            "pthread_attr_getstack" if ecx.frame_in_std() => {
                // We don't support "pthread_attr_setstack", so we just pretend all stacks have the same values here.
                // Hence we can mostly ignore the input `attr_place`.
                let [attr_place, addr_place, size_place] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*libc::pthread_attr_t, *usize, *usize) -> i32),
                    (link_name, abi, args),
                )?;
                let _attr_place = ecx.deref_pointer(attr_place)?;
                let addr_place = ecx.deref_pointer(addr_place)?;
                let size_place = ecx.deref_pointer(size_place)?;

                ecx.write_scalar(
                    Scalar::from_uint(ecx.machine.stack_addr, ecx.pointer_size()),
                    &addr_place,
                )?;
                ecx.write_scalar(
                    Scalar::from_uint(ecx.machine.stack_size, ecx.pointer_size()),
                    &size_place,
                )?;

                // Return success (`0`).
                ecx.write_int(0, dest)?;
            }

            "signal" if ecx.frame_in_std() => {
                let [_, _] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, libc::sighandler_t) -> libc::sighandler_t),
                    (link_name, abi, args),
                )?;
                ecx.write_int(0, dest)?;
            }
            "sigaltstack" if ecx.frame_in_std() => {
                let [_, _] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(*_, *_) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.write_int(0, dest)?;
            }
            "sigaction" if ecx.frame_in_std() => {
                let [_, _, _] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(i32, *_, *_) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.write_int(0, dest)?;
            }

            "getpwuid_r" | "__posix_getpwuid_r" if ecx.frame_in_std() => {
                // getpwuid_r is the standard name, __posix_getpwuid_r is used on solarish
                let [uid, pwd, buf, buflen, result] = &ecx.check_shim_sig(
                    shim_sig!(extern "C" fn(libc::uid_t, *libc::passwd, *_, usize, **_) -> i32),
                    (link_name, abi, args),
                )?;
                ecx.check_no_isolation("`getpwuid_r`")?;

                let uid = ecx.read_scalar(uid)?.to_u32()?;
                let pwd = ecx.deref_pointer(pwd)?;
                let buf = ecx.read_pointer(buf)?;
                let buflen = ecx.read_target_usize(buflen)?;
                let result = ecx.deref_pointer(result)?;

                // Must be for "us".
                if uid != UID {
                    throw_unsup_format!("`getpwuid_r` on other users is not supported");
                }

                // Reset all fields to `uninit` to make sure nobody reads them.
                // (This is a std-only shim so we are okay with such hacks.)
                ecx.write_uninit(&pwd)?;

                // We only set the home_dir field.
                #[allow(deprecated)]
                let home_dir = std::env::home_dir().unwrap();
                let (written, _) = ecx.write_path_to_c_str(&home_dir, buf, buflen)?;
                let pw_dir = ecx.project_field_named(&pwd, "pw_dir")?;
                ecx.write_pointer(buf, &pw_dir)?;

                if written {
                    ecx.write_pointer(pwd.ptr(), &result)?;
                    ecx.write_int(0, dest)?;
                } else {
                    ecx.write_int(0, &result)?;
                    ecx.write_scalar(ecx.eval_libc("ERANGE"), dest)?;
                }
            }

            // Platform-specific shims
            _ => {
                let target_os = &ecx.tcx.sess.target.os;
                return match target_os {
                    Os::Android =>
                        android::EvalContextExt::emulate_foreign_item_inner(
                            ecx, link_name, abi, args, dest,
                        ),
                    Os::FreeBsd =>
                        freebsd::EvalContextExt::emulate_foreign_item_inner(
                            ecx, link_name, abi, args, dest,
                        ),
                    Os::Linux =>
                        linux::EvalContextExt::emulate_foreign_item_inner(
                            ecx, link_name, abi, args, dest,
                        ),
                    Os::MacOs =>
                        macos::EvalContextExt::emulate_foreign_item_inner(
                            ecx, link_name, abi, args, dest,
                        ),
                    Os::Solaris | Os::Illumos =>
                        solarish::EvalContextExt::emulate_foreign_item_inner(
                            ecx, link_name, abi, args, dest,
                        ),
                    Os::NetBsd =>
                        netbsd::EvalContextExt::emulate_foreign_item_inner(
                            ecx, link_name, abi, args, dest,
                        ),
                    _ => interp_ok(EmulateItemResult::NotSupported),
                };
            }
        };

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
