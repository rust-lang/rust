// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::arch::*;
use super::data::{Stat, StatVfs, TimeSpec};
use super::error::Result;
use super::number::*;

use core::mem;

/// Set the end of the process's heap
///
/// When `addr` is `0`, this function will return the current break.
///
/// When `addr` is nonzero, this function will attempt to set the end of the process's
/// heap to `addr` and return the new program break. The new program break should be
/// checked by the allocator, it may not be exactly `addr`, as it may be aligned to a page
/// boundary.
///
/// On error, `Err(ENOMEM)` will be returned indicating that no memory is available
pub unsafe fn brk(addr: usize) -> Result<usize> {
    syscall1(SYS_BRK, addr)
}

/// Change the process's working directory
///
/// This function will attempt to set the process's working directory to `path`, which can be
/// either a relative, scheme relative, or absolute path.
///
/// On success, `Ok(0)` will be returned. On error, one of the following errors will be returned.
///
/// # Errors
///
/// * `EACCES` - permission is denied for one of the components of `path`, or `path`
/// * `EFAULT` - `path` does not point to the process's addressible memory
/// * `EIO` - an I/O error occured
/// * `ENOENT` - `path` does not exit
/// * `ENOTDIR` - `path` is not a directory
pub fn chdir(path: &str) -> Result<usize> {
    unsafe { syscall2(SYS_CHDIR, path.as_ptr() as usize, path.len()) }
}

pub fn chmod(path: &str, mode: usize) -> Result<usize> {
    unsafe { syscall3(SYS_CHMOD, path.as_ptr() as usize, path.len(), mode) }
}

/// Produce a fork of the current process, or a new process thread
pub unsafe fn clone(flags: usize) -> Result<usize> {
    syscall1_clobber(SYS_CLONE, flags)
}

/// Close a file
pub fn close(fd: usize) -> Result<usize> {
    unsafe { syscall1(SYS_CLOSE, fd) }
}

/// Get the current system time
pub fn clock_gettime(clock: usize, tp: &mut TimeSpec) -> Result<usize> {
    unsafe { syscall2(SYS_CLOCK_GETTIME, clock, tp as *mut TimeSpec as usize) }
}

/// Copy and transform a file descriptor
pub fn dup(fd: usize, buf: &[u8]) -> Result<usize> {
    unsafe { syscall3(SYS_DUP, fd, buf.as_ptr() as usize, buf.len()) }
}

/// Replace the current process with a new executable
pub fn execve(path: &str, args: &[[usize; 2]]) -> Result<usize> {
    unsafe { syscall4(SYS_EXECVE, path.as_ptr() as usize, path.len(),
                                  args.as_ptr() as usize, args.len()) }
}

/// Exit the current process
pub fn exit(status: usize) -> Result<usize> {
    unsafe { syscall1(SYS_EXIT, status) }
}

/// Register a file for event-based I/O
pub fn fcntl(fd: usize, cmd: usize, arg: usize) -> Result<usize> {
    unsafe { syscall3(SYS_FCNTL, fd, cmd, arg) }
}

/// Register a file for event-based I/O
pub fn fevent(fd: usize, flags: usize) -> Result<usize> {
    unsafe { syscall2(SYS_FEVENT, fd, flags) }
}

/// Map a file into memory
pub unsafe fn fmap(fd: usize, offset: usize, size: usize) -> Result<usize> {
    syscall3(SYS_FMAP, fd, offset, size)
}

/// Unmap a memory-mapped file
pub unsafe fn funmap(addr: usize) -> Result<usize> {
    syscall1(SYS_FUNMAP, addr)
}

/// Retrieve the canonical path of a file
pub fn fpath(fd: usize, buf: &mut [u8]) -> Result<usize> {
    unsafe { syscall3(SYS_FPATH, fd, buf.as_mut_ptr() as usize, buf.len()) }
}

/// Get metadata about a file
pub fn fstat(fd: usize, stat: &mut Stat) -> Result<usize> {
    unsafe { syscall3(SYS_FSTAT, fd, stat as *mut Stat as usize, mem::size_of::<Stat>()) }
}

/// Get metadata about a filesystem
pub fn fstatvfs(fd: usize, stat: &mut StatVfs) -> Result<usize> {
    unsafe { syscall3(SYS_FSTATVFS, fd, stat as *mut StatVfs as usize, mem::size_of::<StatVfs>()) }
}

/// Sync a file descriptor to its underlying medium
pub fn fsync(fd: usize) -> Result<usize> {
    unsafe { syscall1(SYS_FSYNC, fd) }
}

/// Truncate or extend a file to a specified length
pub fn ftruncate(fd: usize, len: usize) -> Result<usize> {
    unsafe { syscall2(SYS_FTRUNCATE, fd, len) }
}

/// Fast userspace mutex
pub unsafe fn futex(addr: *mut i32, op: usize, val: i32, val2: usize, addr2: *mut i32)
                    -> Result<usize> {
    syscall5(SYS_FUTEX, addr as usize, op, (val as isize) as usize, val2, addr2 as usize)
}

/// Get the current working directory
pub fn getcwd(buf: &mut [u8]) -> Result<usize> {
    unsafe { syscall2(SYS_GETCWD, buf.as_mut_ptr() as usize, buf.len()) }
}

/// Get the effective group ID
pub fn getegid() -> Result<usize> {
    unsafe { syscall0(SYS_GETEGID) }
}

/// Get the effective namespace
pub fn getens() -> Result<usize> {
    unsafe { syscall0(SYS_GETENS) }
}

/// Get the effective user ID
pub fn geteuid() -> Result<usize> {
    unsafe { syscall0(SYS_GETEUID) }
}

/// Get the current group ID
pub fn getgid() -> Result<usize> {
    unsafe { syscall0(SYS_GETGID) }
}

/// Get the current namespace
pub fn getns() -> Result<usize> {
    unsafe { syscall0(SYS_GETNS) }
}

/// Get the current process ID
pub fn getpid() -> Result<usize> {
    unsafe { syscall0(SYS_GETPID) }
}

/// Get the current user ID
pub fn getuid() -> Result<usize> {
    unsafe { syscall0(SYS_GETUID) }
}

/// Set the I/O privilege level
pub unsafe fn iopl(level: usize) -> Result<usize> {
    syscall1(SYS_IOPL, level)
}

/// Send a signal `sig` to the process identified by `pid`
pub fn kill(pid: usize, sig: usize) -> Result<usize> {
    unsafe { syscall2(SYS_KILL, pid, sig) }
}

/// Create a link to a file
pub unsafe fn link(old: *const u8, new: *const u8) -> Result<usize> {
    syscall2(SYS_LINK, old as usize, new as usize)
}

/// Seek to `offset` bytes in a file descriptor
pub fn lseek(fd: usize, offset: isize, whence: usize) -> Result<usize> {
    unsafe { syscall3(SYS_LSEEK, fd, offset as usize, whence) }
}

/// Make a new scheme namespace
pub fn mkns(schemes: &[[usize; 2]]) -> Result<usize> {
    unsafe { syscall2(SYS_MKNS, schemes.as_ptr() as usize, schemes.len()) }
}

/// Sleep for the time specified in `req`
pub fn nanosleep(req: &TimeSpec, rem: &mut TimeSpec) -> Result<usize> {
    unsafe { syscall2(SYS_NANOSLEEP, req as *const TimeSpec as usize,
                                     rem as *mut TimeSpec as usize) }
}

/// Open a file
pub fn open(path: &str, flags: usize) -> Result<usize> {
    unsafe { syscall3(SYS_OPEN, path.as_ptr() as usize, path.len(), flags) }
}

/// Allocate pages, linearly in physical memory
pub unsafe fn physalloc(size: usize) -> Result<usize> {
    syscall1(SYS_PHYSALLOC, size)
}

/// Free physically allocated pages
pub unsafe fn physfree(physical_address: usize, size: usize) -> Result<usize> {
    syscall2(SYS_PHYSFREE, physical_address, size)
}

/// Map physical memory to virtual memory
pub unsafe fn physmap(physical_address: usize, size: usize, flags: usize) -> Result<usize> {
    syscall3(SYS_PHYSMAP, physical_address, size, flags)
}

/// Unmap previously mapped physical memory
pub unsafe fn physunmap(virtual_address: usize) -> Result<usize> {
    syscall1(SYS_PHYSUNMAP, virtual_address)
}

/// Create a pair of file descriptors referencing the read and write ends of a pipe
pub fn pipe2(fds: &mut [usize; 2], flags: usize) -> Result<usize> {
    unsafe { syscall2(SYS_PIPE2, fds.as_ptr() as usize, flags) }
}

/// Read from a file descriptor into a buffer
pub fn read(fd: usize, buf: &mut [u8]) -> Result<usize> {
    unsafe { syscall3(SYS_READ, fd, buf.as_mut_ptr() as usize, buf.len()) }
}

/// Remove a directory
pub fn rmdir(path: &str) -> Result<usize> {
    unsafe { syscall2(SYS_RMDIR, path.as_ptr() as usize, path.len()) }
}

/// Set the current process group IDs
pub fn setregid(rgid: usize, egid: usize) -> Result<usize> {
    unsafe { syscall2(SYS_SETREGID, rgid, egid) }
}

/// Make a new scheme namespace
pub fn setrens(rns: usize, ens: usize) -> Result<usize> {
    unsafe { syscall2(SYS_SETRENS, rns, ens) }
}

/// Set the current process user IDs
pub fn setreuid(ruid: usize, euid: usize) -> Result<usize> {
    unsafe { syscall2(SYS_SETREUID, ruid, euid) }
}

/// Remove a file
pub fn unlink(path: &str) -> Result<usize> {
    unsafe { syscall2(SYS_UNLINK, path.as_ptr() as usize, path.len()) }
}

/// Convert a virtual address to a physical one
pub unsafe fn virttophys(virtual_address: usize) -> Result<usize> {
    syscall1(SYS_VIRTTOPHYS, virtual_address)
}

/// Check if a child process has exited or received a signal
pub fn waitpid(pid: usize, status: &mut usize, options: usize) -> Result<usize> {
    unsafe { syscall3(SYS_WAITPID, pid, status as *mut usize as usize, options) }
}

/// Write a buffer to a file descriptor
///
/// The kernel will attempt to write the bytes in `buf` to the file descriptor `fd`, returning
/// either an `Err`, explained below, or `Ok(count)` where `count` is the number of bytes which
/// were written.
///
/// # Errors
///
/// * `EAGAIN` - the file descriptor was opened with `O_NONBLOCK` and writing would block
/// * `EBADF` - the file descriptor is not valid or is not open for writing
/// * `EFAULT` - `buf` does not point to the process's addressible memory
/// * `EIO` - an I/O error occured
/// * `ENOSPC` - the device containing the file descriptor has no room for data
/// * `EPIPE` - the file descriptor refers to a pipe or socket whose reading end is closed
pub fn write(fd: usize, buf: &[u8]) -> Result<usize> {
    unsafe { syscall3(SYS_WRITE, fd, buf.as_ptr() as usize, buf.len()) }
}

/// Yield the process's time slice to the kernel
///
/// This function will return Ok(0) on success
pub fn sched_yield() -> Result<usize> {
    unsafe { syscall0(SYS_YIELD) }
}
