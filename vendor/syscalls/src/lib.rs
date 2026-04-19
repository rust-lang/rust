#![deny(clippy::all, clippy::pedantic)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::inline_always,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::ptr_as_ptr,
    clippy::unsafe_derive_deserialize
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    // These architectures require nightly to use inline assembly.
    // See https://github.com/rust-lang/rust/issues/93335
    any(
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "s390x",
        target_arch = "powerpc",
        target_arch = "powerpc64",
    ),
    feature(asm_experimental_arch)
)]
#![cfg_attr(doc_cfg, feature(doc_cfg))]

#[macro_use]
mod macros;

mod arch;
mod args;
mod errno;
mod map;
mod set;
mod syscall;

pub use arch::*;
pub use args::SyscallArgs;
pub use errno::{Errno, ErrnoSentinel};
pub use map::*;
pub use set::*;

pub mod raw {
    //! Exposes raw syscalls that simply return a `usize` instead of a `Result`.

    pub use super::syscall::syscall0;
    pub use super::syscall::syscall1;
    pub use super::syscall::syscall2;
    pub use super::syscall::syscall3;
    pub use super::syscall::syscall4;
    pub use super::syscall::syscall5;
    pub use super::syscall::syscall6;
}

/// Issues a system call with 0 arguments.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall0(nr: Sysno) -> Result<usize, Errno> {
    Errno::from_ret(raw::syscall0(nr))
}

/// Issues a system call with 1 argument.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall1(nr: Sysno, a1: usize) -> Result<usize, Errno> {
    Errno::from_ret(raw::syscall1(nr, a1))
}

/// Issues a system call with 2 arguments.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall2(
    nr: Sysno,
    a1: usize,
    a2: usize,
) -> Result<usize, Errno> {
    Errno::from_ret(raw::syscall2(nr, a1, a2))
}

/// Issues a system call with 3 arguments.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall3(
    nr: Sysno,
    a1: usize,
    a2: usize,
    a3: usize,
) -> Result<usize, Errno> {
    Errno::from_ret(raw::syscall3(nr, a1, a2, a3))
}

/// Issues a system call with 4 arguments.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall4(
    nr: Sysno,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
) -> Result<usize, Errno> {
    Errno::from_ret(raw::syscall4(nr, a1, a2, a3, a4))
}

/// Issues a system call with 5 arguments.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall5(
    nr: Sysno,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
) -> Result<usize, Errno> {
    Errno::from_ret(raw::syscall5(nr, a1, a2, a3, a4, a5))
}

/// Issues a system call with 6 arguments.
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
#[inline]
pub unsafe fn syscall6(
    nr: Sysno,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
    a6: usize,
) -> Result<usize, Errno> {
    Errno::from_ret(raw::syscall6(nr, a1, a2, a3, a4, a5, a6))
}

/// Does a raw syscall.
///
/// # Arguments
///  - `nr`: The syscall number.
///  - `args`: packed arguments
///
/// # Returns
///  - `Ok` on success,
///  - `Err` when the syscall failed (with errno).
///
/// # Safety
///
/// Running a system call is inherently unsafe. It is the caller's
/// responsibility to ensure safety.
pub unsafe fn syscall(nr: Sysno, args: &SyscallArgs) -> Result<usize, Errno> {
    syscall6(
        nr, args.arg0, args.arg1, args.arg2, args.arg3, args.arg4, args.arg5,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syscall1_syscall4() {
        let fd = unsafe {
            let at_fdcwd = -100isize;
            syscall!(Sysno::openat, at_fdcwd, "/dev/zero\0".as_ptr(), 0)
        }
        .unwrap();

        let mut buffer1: [u8; 64] = unsafe { core::mem::zeroed() };
        let mut buffer2: [u8; 64] = unsafe { core::mem::zeroed() };

        let r1 =
            unsafe { libc::read(fd as i32, buffer1.as_mut_ptr() as _, 64) };

        let s1 = unsafe {
            core::slice::from_raw_parts(
                buffer1.as_mut_ptr() as *const u8,
                r1 as usize,
            )
        };
        let r2 = unsafe { syscall!(Sysno::read, fd, buffer2.as_mut_ptr(), 64) };
        let s2 = unsafe {
            core::slice::from_raw_parts(
                buffer1.as_mut_ptr() as *const u8,
                r2.unwrap_or(0),
            )
        };

        assert_eq!(r2, Ok(r1 as usize));
        assert_eq!(s1, s2);

        let closed = unsafe { syscall!(Sysno::close, fd) };
        assert!(closed.is_ok());
    }

    #[test]
    fn test_syscall1_syscall4_2() {
        let fd = unsafe {
            let at_fdcwd = -100isize;
            syscall!(Sysno::openat, at_fdcwd, "/dev/zero\0".as_ptr(), 0)
        }
        .unwrap();

        let mut buffer1: [u8; 64] = unsafe { core::mem::zeroed() };
        let mut buffer2: [u8; 64] = unsafe { core::mem::zeroed() };

        let args = SyscallArgs::from(&[fd, buffer1.as_mut_ptr() as _, 64]);
        let r1 = unsafe { syscall(Sysno::read, &args) }.expect("read failed");

        let s1 = unsafe {
            core::slice::from_raw_parts(buffer1.as_mut_ptr() as *const u8, r1)
        };
        let r2 = unsafe { syscall!(Sysno::read, fd, buffer2.as_mut_ptr(), 64) };
        let s2 = unsafe {
            core::slice::from_raw_parts(
                buffer1.as_mut_ptr() as *const u8,
                r2.unwrap_or(0),
            )
        };

        assert_eq!(r2, Ok(r1));
        assert_eq!(s1, s2);

        let closed = unsafe { syscall!(Sysno::close, fd) };
        assert!(closed.is_ok());
    }

    #[test]
    fn test_name() {
        assert_eq!(Sysno::write.name(), "write");
        assert_eq!(Sysno::fsopen.name(), "fsopen");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_syscallno() {
        assert_eq!(Sysno::from(2), Sysno::open);
        assert_eq!(Sysno::new(2), Some(Sysno::open));
        assert_eq!(Sysno::new(-1i32 as usize), None);
        assert_eq!(Sysno::new(1024), None);
    }

    #[test]
    fn test_first() {
        #[cfg(target_arch = "x86_64")]
        assert_eq!(Sysno::first(), Sysno::read);

        #[cfg(target_arch = "x86")]
        assert_eq!(Sysno::first(), Sysno::restart_syscall);
    }

    #[test]
    fn test_syscall_len() {
        assert!(Sysno::table_size() > 300);
        assert!(Sysno::table_size() < 1000);
    }
}
