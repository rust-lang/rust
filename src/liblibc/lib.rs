// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "libc"]
#![crate_type = "rlib"]
#![cfg_attr(not(feature = "cargo-build"),
            unstable(feature = "libc"))]
#![cfg_attr(not(feature = "cargo-build"), feature(staged_api))]
#![cfg_attr(not(feature = "cargo-build"), staged_api)]
#![cfg_attr(not(feature = "cargo-build"), feature(core))]
#![feature(no_std)]
#![no_std]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/nightly/",
       html_playground_url = "http://play.rust-lang.org/")]
#![cfg_attr(test, feature(test))]

//! Bindings for the C standard library and other platform libraries
//!
//! **NOTE:** These are *architecture and libc* specific. On Linux, these
//! bindings are only correct for glibc.
//!
//! This module contains bindings to the C standard library, organized into
//! modules by their defining standard.  Additionally, it contains some assorted
//! platform-specific definitions.  For convenience, most functions and types
//! are reexported, so `use libc::*` will import the available C bindings as
//! appropriate for the target platform. The exact set of functions available
//! are platform specific.
//!
//! *Note:* Because these definitions are platform-specific, some may not appear
//! in the generated documentation.
//!
//! We consider the following specs reasonably normative with respect to
//! interoperating with the C standard library (libc/msvcrt):
//!
//! * ISO 9899:1990 ('C95', 'ANSI C', 'Standard C'), NA1, 1995.
//! * ISO 9899:1999 ('C99' or 'C9x').
//! * ISO 9945:1988 / IEEE 1003.1-1988 ('POSIX.1').
//! * ISO 9945:2001 / IEEE 1003.1-2001 ('POSIX:2001', 'SUSv3').
//! * ISO 9945:2008 / IEEE 1003.1-2008 ('POSIX:2008', 'SUSv4').
//!
//! Note that any reference to the 1996 revision of POSIX, or any revs between
//! 1990 (when '88 was approved at ISO) and 2001 (when the next actual
//! revision-revision happened), are merely additions of other chapters (1b and
//! 1c) outside the core interfaces.
//!
//! Despite having several names each, these are *reasonably* coherent
//! point-in-time, list-of-definition sorts of specs. You can get each under a
//! variety of names but will wind up with the same definition in each case.
//!
//! See standards(7) in linux-manpages for more details.
//!
//! Our interface to these libraries is complicated by the non-universality of
//! conformance to any of them. About the only thing universally supported is
//! the first (C95), beyond that definitions quickly become absent on various
//! platforms.
//!
//! We therefore wind up dividing our module-space up (mostly for the sake of
//! sanity while editing, filling-in-details and eliminating duplication) into
//! definitions common-to-all (held in modules named c95, c99, posix88, posix01
//! and posix08) and definitions that appear only on *some* platforms (named
//! 'extra'). This would be things like significant OSX foundation kit, or Windows
//! library kernel32.dll, or various fancy glibc, Linux or BSD extensions.
//!
//! In addition to the per-platform 'extra' modules, we define a module of
//! 'common BSD' libc routines that never quite made it into POSIX but show up
//! in multiple derived systems. This is the 4.4BSD r2 / 1995 release, the final
//! one from Berkeley after the lawsuits died down and the CSRG dissolved.

#![allow(bad_style, raw_pointer_derive)]
#[cfg(feature = "cargo-build")] extern crate "std" as core;
#[cfg(not(feature = "cargo-build"))] extern crate core;

#[cfg(test)] extern crate std;
#[cfg(test)] extern crate test;

// Explicit export lists for the intersection (provided here) mean that
// you can write more-platform-agnostic code if you stick to just these
// symbols.

pub use types::common::c95::{FILE, c_void, fpos_t};
pub use types::common::c99::{int8_t, int16_t, int32_t, int64_t};
pub use types::common::c99::{uint8_t, uint16_t, uint32_t, uint64_t};
pub use types::common::posix88::{DIR, dirent_t};
pub use types::os::common::posix01::{timeval};
pub use types::os::common::bsd44::{addrinfo, in_addr, in6_addr, sockaddr_storage};
pub use types::os::common::bsd44::{ip_mreq, ip6_mreq, sockaddr, sockaddr_un};
pub use types::os::common::bsd44::{sa_family_t, sockaddr_in, sockaddr_in6, socklen_t};
pub use types::os::arch::c95::{c_char, c_double, c_float, c_int, c_uint};
pub use types::os::arch::c95::{c_long, c_short, c_uchar, c_ulong, wchar_t};
pub use types::os::arch::c95::{c_ushort, clock_t, ptrdiff_t, c_schar};
pub use types::os::arch::c95::{size_t, time_t, suseconds_t};
pub use types::os::arch::c99::{c_longlong, c_ulonglong};
pub use types::os::arch::c99::{intptr_t, uintptr_t};
pub use types::os::arch::c99::{intmax_t, uintmax_t};
pub use types::os::arch::posix88::{dev_t, ino_t, mode_t};
pub use types::os::arch::posix88::{off_t, pid_t, ssize_t};

pub use consts::os::c95::{_IOFBF, _IOLBF, _IONBF, BUFSIZ, EOF};
pub use consts::os::c95::{EXIT_FAILURE, EXIT_SUCCESS};
pub use consts::os::c95::{FILENAME_MAX, FOPEN_MAX, L_tmpnam};
pub use consts::os::c95::{RAND_MAX, SEEK_CUR, SEEK_END};
pub use consts::os::c95::{SEEK_SET, TMP_MAX};
pub use consts::os::posix88::{F_OK, O_APPEND, O_CREAT, O_EXCL};
pub use consts::os::posix88::{O_RDONLY, O_RDWR, O_TRUNC, O_WRONLY};
pub use consts::os::posix88::{R_OK, S_IEXEC, S_IFBLK, S_IFCHR};
pub use consts::os::posix88::{S_IFDIR, S_IFIFO, S_IFMT, S_IFREG, S_IFLNK};
pub use consts::os::posix88::{S_IREAD, S_IRUSR, S_IRWXU, S_IWUSR};
pub use consts::os::posix88::{STDERR_FILENO, STDIN_FILENO, S_IXUSR};
pub use consts::os::posix88::{STDOUT_FILENO, W_OK, X_OK};
pub use consts::os::bsd44::{AF_INET, AF_INET6, SOCK_STREAM, SOCK_DGRAM, SOCK_RAW};
pub use consts::os::bsd44::{IPPROTO_IP, IPPROTO_IPV6, IPPROTO_TCP, TCP_NODELAY};
pub use consts::os::bsd44::{SOL_SOCKET, SO_KEEPALIVE, SO_ERROR};
pub use consts::os::bsd44::{SO_REUSEADDR, SO_BROADCAST, SHUT_WR, IP_MULTICAST_LOOP};
pub use consts::os::bsd44::{IP_ADD_MEMBERSHIP, IP_DROP_MEMBERSHIP};
pub use consts::os::bsd44::{IPV6_ADD_MEMBERSHIP, IPV6_DROP_MEMBERSHIP};
pub use consts::os::bsd44::{IP_MULTICAST_TTL, IP_TTL, IP_HDRINCL, SHUT_RD};
pub use consts::os::extra::{IPPROTO_RAW};

pub use funcs::c95::ctype::{isalnum, isalpha, iscntrl, isdigit};
pub use funcs::c95::ctype::{islower, isprint, ispunct, isspace};
pub use funcs::c95::ctype::{isupper, isxdigit, tolower, toupper};

pub use funcs::c95::stdio::{fclose, feof, ferror, fflush, fgetc};
pub use funcs::c95::stdio::{fgetpos, fgets, fopen, fputc, fputs};
pub use funcs::c95::stdio::{fread, freopen, fseek, fsetpos, ftell};
pub use funcs::c95::stdio::{fwrite, perror, puts, remove, rename, rewind};
pub use funcs::c95::stdio::{setbuf, setvbuf, tmpfile, ungetc};

pub use funcs::c95::stdlib::{abs, atof, atoi, calloc, exit, _exit, atexit};
pub use funcs::c95::stdlib::{free, getenv, labs, malloc, rand};
pub use funcs::c95::stdlib::{realloc, srand, strtod, strtol};
pub use funcs::c95::stdlib::{strtoul, system};

pub use funcs::c95::string::{memchr, memcmp};
pub use funcs::c95::string::{strcat, strchr, strcmp};
pub use funcs::c95::string::{strcoll, strcpy, strcspn, strerror};
pub use funcs::c95::string::{strlen, strncat, strncmp, strncpy};
pub use funcs::c95::string::{strpbrk, strrchr, strspn, strstr};
pub use funcs::c95::string::{strtok, strxfrm};

pub use funcs::posix88::fcntl::{open, creat};
pub use funcs::posix88::stat_::{chmod, fstat, mkdir, stat};
pub use funcs::posix88::stdio::{fdopen, fileno, pclose, popen};
pub use funcs::posix88::unistd::{access, chdir, close, dup, dup2};
pub use funcs::posix88::unistd::{execv, execve, execvp, getcwd};
pub use funcs::posix88::unistd::{getpid, isatty, lseek, pipe, read};
pub use funcs::posix88::unistd::{rmdir, unlink, write};

pub use funcs::bsd43::{socket, setsockopt, bind, send, recv, recvfrom};
pub use funcs::bsd43::{listen, sendto, accept, connect, getpeername, getsockname};
pub use funcs::bsd43::{shutdown};

// But we also reexport most everything
// if you're interested in writing platform-specific code.

// FIXME: This is a mess, but the design of this entire module needs to be
// reconsidered, so I'm not inclined to do better right now. As part of
// #11870 I removed all the pub globs here, leaving explicit reexports
// of everything that is actually used in-tree.
//
// So the following exports don't follow any particular plan.

#[cfg(unix)] pub use consts::os::sysconf::{_SC_PAGESIZE};
#[cfg(unix)] pub use consts::os::posix88::{PROT_READ, PROT_WRITE, PROT_EXEC};
#[cfg(unix)] pub use consts::os::posix88::{MAP_FIXED, MAP_FILE, MAP_ANON, MAP_PRIVATE, MAP_FAILED};
#[cfg(unix)] pub use consts::os::posix88::{EACCES, EBADF, EINVAL, ENODEV, ENOMEM};
#[cfg(unix)] pub use consts::os::posix88::{ECONNREFUSED, ECONNRESET, EPERM, EPIPE};
#[cfg(unix)] pub use consts::os::posix88::{ENOTCONN, ECONNABORTED, EADDRNOTAVAIL, EINTR};
#[cfg(unix)] pub use consts::os::posix88::{EADDRINUSE, ENOENT, EISDIR, EAGAIN, EWOULDBLOCK};
#[cfg(unix)] pub use consts::os::posix88::{ECANCELED, SIGINT, EINPROGRESS};
#[cfg(unix)] pub use consts::os::posix88::{ENOSYS, ENOTTY, ETIMEDOUT, EMFILE};
#[cfg(unix)] pub use consts::os::posix88::{SIGTERM, SIGKILL, SIGPIPE, PROT_NONE};
#[cfg(unix)] pub use consts::os::posix01::{SIG_IGN, F_GETFL, F_SETFL};
#[cfg(unix)] pub use consts::os::bsd44::{AF_UNIX};
#[cfg(unix)] pub use consts::os::extra::{O_NONBLOCK};

#[cfg(unix)] pub use types::os::common::posix01::{pthread_t, timespec, timezone};

#[cfg(unix)] pub use types::os::arch::posix88::{uid_t, gid_t};
#[cfg(unix)] pub use types::os::arch::posix01::{pthread_attr_t};
#[cfg(unix)] pub use types::os::arch::posix01::{stat, utimbuf};
#[cfg(unix)] pub use types::os::common::bsd44::{ifaddrs};
#[cfg(unix)] pub use funcs::posix88::unistd::{sysconf, setgid, setsid, setuid, pread, pwrite};
#[cfg(unix)] pub use funcs::posix88::unistd::{getgid, getuid, getsid};
#[cfg(unix)] pub use funcs::posix88::unistd::{_PC_NAME_MAX, utime, nanosleep, pathconf, link};
#[cfg(unix)] pub use funcs::posix88::unistd::{chown};
#[cfg(unix)] pub use funcs::posix88::mman::{mmap, munmap, mprotect};
#[cfg(unix)] pub use funcs::posix88::dirent::{opendir, readdir_r, closedir};
#[cfg(unix)] pub use funcs::posix88::fcntl::{fcntl};
#[cfg(unix)] pub use funcs::posix88::net::{if_nametoindex};
#[cfg(unix)] pub use funcs::posix01::stat_::{lstat};
#[cfg(unix)] pub use funcs::posix01::unistd::{fsync, ftruncate};
#[cfg(unix)] pub use funcs::posix01::unistd::{readlink, symlink};
#[cfg(unix)] pub use funcs::bsd43::{getifaddrs, freeifaddrs};

#[cfg(windows)] pub use consts::os::c95::{WSAECONNREFUSED, WSAECONNRESET, WSAEACCES};
#[cfg(windows)] pub use consts::os::c95::{WSAEWOULDBLOCK, WSAENOTCONN, WSAECONNABORTED};
#[cfg(windows)] pub use consts::os::c95::{WSAEADDRNOTAVAIL, WSAEADDRINUSE, WSAEINTR};
#[cfg(windows)] pub use consts::os::c95::{WSAEINPROGRESS, WSAEINVAL, WSAEMFILE};
#[cfg(windows)] pub use consts::os::extra::{ERROR_INSUFFICIENT_BUFFER};
#[cfg(windows)] pub use consts::os::extra::{O_BINARY, O_NOINHERIT, PAGE_NOACCESS};
#[cfg(windows)] pub use consts::os::extra::{PAGE_READONLY, PAGE_READWRITE, PAGE_EXECUTE};
#[cfg(windows)] pub use consts::os::extra::{PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE};
#[cfg(windows)] pub use consts::os::extra::{MEM_COMMIT, MEM_RESERVE, MEM_RELEASE};
#[cfg(windows)] pub use consts::os::extra::{FILE_MAP_READ, FILE_MAP_WRITE, FILE_MAP_EXECUTE};
#[cfg(windows)] pub use consts::os::extra::{ERROR_ALREADY_EXISTS, ERROR_NO_DATA};
#[cfg(windows)] pub use consts::os::extra::{ERROR_FILE_NOT_FOUND, ERROR_INVALID_NAME};
#[cfg(windows)] pub use consts::os::extra::{ERROR_BROKEN_PIPE, ERROR_INVALID_FUNCTION};
#[cfg(windows)] pub use consts::os::extra::{ERROR_CALL_NOT_IMPLEMENTED};
#[cfg(windows)] pub use consts::os::extra::{ERROR_NOTHING_TO_TERMINATE};
#[cfg(windows)] pub use consts::os::extra::{ERROR_INVALID_HANDLE};
#[cfg(windows)] pub use consts::os::extra::{TRUE, FALSE, INFINITE};
#[cfg(windows)] pub use consts::os::extra::{PROCESS_TERMINATE, PROCESS_QUERY_INFORMATION};
#[cfg(windows)] pub use consts::os::extra::{STILL_ACTIVE, DETACHED_PROCESS};
#[cfg(windows)] pub use consts::os::extra::{CREATE_NEW_PROCESS_GROUP, CREATE_UNICODE_ENVIRONMENT};
#[cfg(windows)] pub use consts::os::extra::{FILE_BEGIN, FILE_END, FILE_CURRENT};
#[cfg(windows)] pub use consts::os::extra::{FILE_GENERIC_READ, FILE_GENERIC_WRITE};
#[cfg(windows)] pub use consts::os::extra::{FILE_SHARE_READ, FILE_SHARE_WRITE, FILE_SHARE_DELETE};
#[cfg(windows)] pub use consts::os::extra::{TRUNCATE_EXISTING, CREATE_ALWAYS, OPEN_EXISTING};
#[cfg(windows)] pub use consts::os::extra::{CREATE_NEW, FILE_APPEND_DATA, FILE_WRITE_DATA};
#[cfg(windows)] pub use consts::os::extra::{OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL};
#[cfg(windows)] pub use consts::os::extra::{FILE_FLAG_BACKUP_SEMANTICS, INVALID_HANDLE_VALUE};
#[cfg(windows)] pub use consts::os::extra::{MOVEFILE_REPLACE_EXISTING};
#[cfg(windows)] pub use consts::os::extra::{GENERIC_READ, GENERIC_WRITE};
#[cfg(windows)] pub use consts::os::extra::{VOLUME_NAME_DOS};
#[cfg(windows)] pub use consts::os::extra::{PIPE_ACCESS_DUPLEX, FILE_FLAG_FIRST_PIPE_INSTANCE};
#[cfg(windows)] pub use consts::os::extra::{FILE_FLAG_OVERLAPPED, PIPE_TYPE_BYTE};
#[cfg(windows)] pub use consts::os::extra::{PIPE_READMODE_BYTE, PIPE_WAIT};
#[cfg(windows)] pub use consts::os::extra::{PIPE_UNLIMITED_INSTANCES, ERROR_ACCESS_DENIED};
#[cfg(windows)] pub use consts::os::extra::{FILE_WRITE_ATTRIBUTES, FILE_READ_ATTRIBUTES};
#[cfg(windows)] pub use consts::os::extra::{ERROR_PIPE_BUSY, ERROR_IO_PENDING};
#[cfg(windows)] pub use consts::os::extra::{ERROR_PIPE_CONNECTED, WAIT_OBJECT_0};
#[cfg(windows)] pub use consts::os::extra::{ERROR_NOT_FOUND};
#[cfg(windows)] pub use consts::os::extra::{ERROR_OPERATION_ABORTED};
#[cfg(windows)] pub use consts::os::extra::{FIONBIO};
#[cfg(windows)] pub use types::os::common::bsd44::{SOCKET};
#[cfg(windows)] pub use types::os::common::posix01::{stat, utimbuf};
#[cfg(windows)] pub use types::os::arch::extra::{HANDLE, BOOL, LPSECURITY_ATTRIBUTES};
#[cfg(windows)] pub use types::os::arch::extra::{LPCSTR, WORD, DWORD, BYTE, FILETIME};
#[cfg(windows)] pub use types::os::arch::extra::{LARGE_INTEGER, LPVOID, LONG};
#[cfg(windows)] pub use types::os::arch::extra::{time64_t, OVERLAPPED, LPCWSTR};
#[cfg(windows)] pub use types::os::arch::extra::{LPOVERLAPPED, SIZE_T, LPDWORD};
#[cfg(windows)] pub use types::os::arch::extra::{SECURITY_ATTRIBUTES, WIN32_FIND_DATAW};
#[cfg(windows)] pub use funcs::c95::string::{wcslen};
#[cfg(windows)] pub use funcs::posix88::stat_::{wstat, wutime, wchmod, wrmdir};
#[cfg(windows)] pub use funcs::bsd43::{closesocket};
#[cfg(windows)] pub use funcs::extra::kernel32::{GetCurrentDirectoryW, GetLastError};
#[cfg(windows)] pub use funcs::extra::kernel32::{GetEnvironmentVariableW, SetEnvironmentVariableW};
#[cfg(windows)] pub use funcs::extra::kernel32::{GetModuleFileNameW, SetCurrentDirectoryW};
#[cfg(windows)] pub use funcs::extra::kernel32::{GetSystemInfo, VirtualAlloc, VirtualFree};
#[cfg(windows)] pub use funcs::extra::kernel32::{CreateFileMappingW, MapViewOfFile};
#[cfg(windows)] pub use funcs::extra::kernel32::{UnmapViewOfFile, CloseHandle};
#[cfg(windows)] pub use funcs::extra::kernel32::{WaitForSingleObject, GetSystemTimeAsFileTime};
#[cfg(windows)] pub use funcs::extra::kernel32::{QueryPerformanceCounter};
#[cfg(windows)] pub use funcs::extra::kernel32::{QueryPerformanceFrequency};
#[cfg(windows)] pub use funcs::extra::kernel32::{GetExitCodeProcess, TerminateProcess};
#[cfg(windows)] pub use funcs::extra::kernel32::{ReadFile, WriteFile, SetFilePointerEx};
#[cfg(windows)] pub use funcs::extra::kernel32::{SetEndOfFile, CreateFileW};
#[cfg(windows)] pub use funcs::extra::kernel32::{CreateDirectoryW, FindFirstFileW};
#[cfg(windows)] pub use funcs::extra::kernel32::{FindNextFileW, FindClose, DeleteFileW};
#[cfg(windows)] pub use funcs::extra::kernel32::{CreateHardLinkW, CreateEventW};
#[cfg(windows)] pub use funcs::extra::kernel32::{FlushFileBuffers, CreateNamedPipeW};
#[cfg(windows)] pub use funcs::extra::kernel32::{SetNamedPipeHandleState, WaitNamedPipeW};
#[cfg(windows)] pub use funcs::extra::kernel32::{GetOverlappedResult, ConnectNamedPipe};
#[cfg(windows)] pub use funcs::extra::kernel32::{DisconnectNamedPipe, OpenProcess};
#[cfg(windows)] pub use funcs::extra::kernel32::{MoveFileExW, VirtualProtect};
#[cfg(windows)] pub use funcs::extra::kernel32::{RemoveDirectoryW};
#[cfg(windows)] pub use funcs::extra::msvcrt::{get_osfhandle, open_osfhandle};
#[cfg(windows)] pub use funcs::extra::winsock::{ioctlsocket};

#[cfg(any(target_os = "linux",
          target_os = "android",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "openbsd"))]
pub use consts::os::posix01::{CLOCK_REALTIME, CLOCK_MONOTONIC};

#[cfg(any(target_os = "linux", target_os = "android"))]
pub use funcs::posix01::unistd::{fdatasync};
#[cfg(any(target_os = "linux", target_os = "android"))]
pub use types::os::arch::extra::{sockaddr_ll};
#[cfg(any(target_os = "linux", target_os = "android"))]
pub use consts::os::extra::{AF_PACKET};

#[cfg(all(unix, not(any(target_os = "freebsd", target_os = "openbsd"))))]
pub use consts::os::extra::{MAP_STACK};

#[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
pub use consts::os::bsd44::{TCP_KEEPIDLE};

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use consts::os::bsd44::{TCP_KEEPALIVE};
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use consts::os::extra::{F_FULLFSYNC};

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use types::os::arch::extra::{mach_timebase_info};


#[cfg(not(windows))]
#[link(name = "c")]
#[link(name = "m")]
extern {}

pub mod types {

    // Types tend to vary *per architecture* so we pull their definitions out
    // into this module.

    // Standard types that are opaque or common, so are not per-target.
    pub mod common {
        pub mod c95 {
            /// Type used to construct void pointers for use with C.
            ///
            /// This type is only useful as a pointer target. Do not use it as a
            /// return type for FFI functions which have the `void` return type in
            /// C. Use the unit type `()` or omit the return type instead.
            ///
            /// For LLVM to recognize the void pointer type and by extension
            /// functions like malloc(), we need to have it represented as i8* in
            /// LLVM bitcode. The enum used here ensures this and prevents misuse
            /// of the "raw" type by only having private variants.. We need two
            /// variants, because the compiler complains about the repr attribute
            /// otherwise.
            #[repr(u8)]
            pub enum c_void {
                __variant1,
                __variant2,
            }

            pub enum FILE {}
            pub enum fpos_t {}
        }
        pub mod c99 {
            pub type int8_t = i8;
            pub type int16_t = i16;
            pub type int32_t = i32;
            pub type int64_t = i64;
            pub type uint8_t = u8;
            pub type uint16_t = u16;
            pub type uint32_t = u32;
            pub type uint64_t = u64;
        }
        pub mod posix88 {
            pub enum DIR {}
            pub enum dirent_t {}
        }
        pub mod posix01 {}
        pub mod posix08 {}
        pub mod bsd44 {}
    }

    // Standard types that are scalar but vary by OS and arch.

    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub mod os {
        pub mod common {
            pub mod posix01 {
                use types::common::c95::{c_void};
                use types::os::arch::c95::{c_char, c_ulong, size_t,
                                                 time_t, suseconds_t, c_long};

                pub type pthread_t = c_ulong;

                #[repr(C)]
                #[derive(Copy)] pub struct glob_t {
                    pub gl_pathc: size_t,
                    pub gl_pathv: *mut *mut c_char,
                    pub gl_offs:  size_t,

                    pub __unused1: *mut c_void,
                    pub __unused2: *mut c_void,
                    pub __unused3: *mut c_void,
                    pub __unused4: *mut c_void,
                    pub __unused5: *mut c_void,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct timeval {
                    pub tv_sec: time_t,
                    pub tv_usec: suseconds_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct timespec {
                    pub tv_sec: time_t,
                    pub tv_nsec: c_long,
                }

                #[derive(Copy)] pub enum timezone {}

                pub type sighandler_t = size_t;
            }
            pub mod bsd44 {
                use types::common::c95::{c_void};
                use types::os::arch::c95::{c_char, c_int, c_uint};

                pub type socklen_t = u32;
                pub type sa_family_t = u16;
                pub type in_port_t = u16;
                pub type in_addr_t = u32;
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr {
                    pub sa_family: sa_family_t,
                    pub sa_data: [u8; 14],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_storage {
                    pub ss_family: sa_family_t,
                    pub __ss_align: i64,
                    pub __ss_pad2: [u8; 112],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_in {
                    pub sin_family: sa_family_t,
                    pub sin_port: in_port_t,
                    pub sin_addr: in_addr,
                    pub sin_zero: [u8; 8],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct in_addr {
                    pub s_addr: in_addr_t,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_in6 {
                    pub sin6_family: sa_family_t,
                    pub sin6_port: in_port_t,
                    pub sin6_flowinfo: u32,
                    pub sin6_addr: in6_addr,
                    pub sin6_scope_id: u32,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct in6_addr {
                    pub s6_addr: [u16; 8]
                }
                #[repr(C)]
                #[derive(Copy)] pub struct ip_mreq {
                    pub imr_multiaddr: in_addr,
                    pub imr_interface: in_addr,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct ip6_mreq {
                    pub ipv6mr_multiaddr: in6_addr,
                    pub ipv6mr_interface: c_uint,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct addrinfo {
                    pub ai_flags: c_int,
                    pub ai_family: c_int,
                    pub ai_socktype: c_int,
                    pub ai_protocol: c_int,
                    pub ai_addrlen: socklen_t,

                    #[cfg(target_os = "linux")]
                    pub ai_addr: *mut sockaddr,

                    #[cfg(target_os = "linux")]
                    pub ai_canonname: *mut c_char,

                    #[cfg(target_os = "android")]
                    pub ai_canonname: *mut c_char,

                    #[cfg(target_os = "android")]
                    pub ai_addr: *mut sockaddr,

                    pub ai_next: *mut addrinfo,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_un {
                    pub sun_family: sa_family_t,
                    pub sun_path: [c_char; 108]
                }

                #[repr(C)]
                #[derive(Copy)] pub struct ifaddrs {
                    pub ifa_next: *mut ifaddrs,
                    pub ifa_name: *mut c_char,
                    pub ifa_flags: c_uint,
                    pub ifa_addr: *mut sockaddr,
                    pub ifa_netmask: *mut sockaddr,
                    pub ifa_ifu: *mut sockaddr, // FIXME This should be a union
                    pub ifa_data: *mut c_void
                }

            }
        }

        #[cfg(any(target_arch = "x86",
                  target_arch = "arm",
                  target_arch = "mips",
                  target_arch = "mipsel",
                  target_arch = "powerpc"))]
        pub mod arch {
            pub mod c95 {
                pub type c_char = i8;
                pub type c_schar = i8;
                pub type c_uchar = u8;
                pub type c_short = i16;
                pub type c_ushort = u16;
                pub type c_int = i32;
                pub type c_uint = u32;
                pub type c_long = i32;
                pub type c_ulong = u32;
                pub type c_float = f32;
                pub type c_double = f64;
                pub type size_t = u32;
                pub type ptrdiff_t = i32;
                pub type clock_t = i32;
                pub type time_t = i32;
                pub type suseconds_t = i32;
                pub type wchar_t = i32;
            }
            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;
                pub type intptr_t = i32;
                pub type uintptr_t = u32;
                pub type intmax_t = i64;
                pub type uintmax_t = u64;
            }
            #[cfg(any(target_arch = "x86",
                      target_arch = "mips",
                      target_arch = "mipsel",
                      target_arch = "powerpc"))]
            pub mod posix88 {
                pub type off_t = i32;
                pub type dev_t = u64;
                pub type ino_t = u32;
                pub type pid_t = i32;
                pub type uid_t = u32;
                pub type gid_t = u32;
                pub type useconds_t = u32;
                pub type mode_t = u32;
                pub type ssize_t = i32;
            }
            #[cfg(target_arch = "arm")]
            pub mod posix88 {
                pub type off_t = i32;
                pub type dev_t = u32;
                pub type ino_t = u32;
                pub type pid_t = i32;
                pub type uid_t = u32;
                pub type gid_t = u32;
                pub type useconds_t = u32;
                pub type mode_t = u16;
                pub type ssize_t = i32;
            }
            #[cfg(any(target_arch = "x86",
                      target_arch = "powerpc"))]
            pub mod posix01 {
                use types::os::arch::c95::{c_short, c_long, time_t};
                use types::os::arch::posix88::{dev_t, gid_t, ino_t};
                use types::os::arch::posix88::{mode_t, off_t};
                use types::os::arch::posix88::{uid_t};

                pub type nlink_t = u32;
                pub type blksize_t = i32;
                pub type blkcnt_t = i32;

                #[repr(C)]
                #[derive(Copy)] pub struct stat {
                    pub st_dev: dev_t,
                    pub __pad1: c_short,
                    pub st_ino: ino_t,
                    pub st_mode: mode_t,
                    pub st_nlink: nlink_t,
                    pub st_uid: uid_t,
                    pub st_gid: gid_t,
                    pub st_rdev: dev_t,
                    pub __pad2: c_short,
                    pub st_size: off_t,
                    pub st_blksize: blksize_t,
                    pub st_blocks: blkcnt_t,
                    pub st_atime: time_t,
                    pub st_atime_nsec: c_long,
                    pub st_mtime: time_t,
                    pub st_mtime_nsec: c_long,
                    pub st_ctime: time_t,
                    pub st_ctime_nsec: c_long,
                    pub __unused4: c_long,
                    pub __unused5: c_long,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct utimbuf {
                    pub actime: time_t,
                    pub modtime: time_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct pthread_attr_t {
                    pub __size: [u32; 9]
                }
            }
            #[cfg(target_arch = "arm")]
            pub mod posix01 {
                use types::os::arch::c95::{c_uchar, c_uint, c_ulong, time_t};
                use types::os::arch::c99::{c_longlong, c_ulonglong};
                use types::os::arch::posix88::{uid_t, gid_t, ino_t};

                pub type nlink_t = u16;
                pub type blksize_t = u32;
                pub type blkcnt_t = u32;

                #[repr(C)]
                #[derive(Copy)] pub struct stat {
                    pub st_dev: c_ulonglong,
                    pub __pad0: [c_uchar; 4],
                    pub __st_ino: ino_t,
                    pub st_mode: c_uint,
                    pub st_nlink: c_uint,
                    pub st_uid: uid_t,
                    pub st_gid: gid_t,
                    pub st_rdev: c_ulonglong,
                    pub __pad3: [c_uchar; 4],
                    pub st_size: c_longlong,
                    pub st_blksize: blksize_t,
                    pub st_blocks: c_ulonglong,
                    pub st_atime: time_t,
                    pub st_atime_nsec: c_ulong,
                    pub st_mtime: time_t,
                    pub st_mtime_nsec: c_ulong,
                    pub st_ctime: time_t,
                    pub st_ctime_nsec: c_ulong,
                    pub st_ino: c_ulonglong,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct utimbuf {
                    pub actime: time_t,
                    pub modtime: time_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct pthread_attr_t {
                    pub __size: [u32; 9]
                }
            }
            #[cfg(any(target_arch = "mips",
                      target_arch = "mipsel"))]
            pub mod posix01 {
                use types::os::arch::c95::{c_long, c_ulong, time_t};
                use types::os::arch::posix88::{gid_t, ino_t};
                use types::os::arch::posix88::{mode_t, off_t};
                use types::os::arch::posix88::{uid_t};

                pub type nlink_t = u32;
                pub type blksize_t = i32;
                pub type blkcnt_t = i32;

                #[repr(C)]
                #[derive(Copy)] pub struct stat {
                    pub st_dev: c_ulong,
                    pub st_pad1: [c_long; 3],
                    pub st_ino: ino_t,
                    pub st_mode: mode_t,
                    pub st_nlink: nlink_t,
                    pub st_uid: uid_t,
                    pub st_gid: gid_t,
                    pub st_rdev: c_ulong,
                    pub st_pad2: [c_long; 2],
                    pub st_size: off_t,
                    pub st_pad3: c_long,
                    pub st_atime: time_t,
                    pub st_atime_nsec: c_long,
                    pub st_mtime: time_t,
                    pub st_mtime_nsec: c_long,
                    pub st_ctime: time_t,
                    pub st_ctime_nsec: c_long,
                    pub st_blksize: blksize_t,
                    pub st_blocks: blkcnt_t,
                    pub st_pad5: [c_long; 14],
                }

                #[repr(C)]
                #[derive(Copy)] pub struct utimbuf {
                    pub actime: time_t,
                    pub modtime: time_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct pthread_attr_t {
                    pub __size: [u32; 9]
                }
            }
            pub mod posix08 {}
            pub mod bsd44 {}
            pub mod extra {
                use types::os::arch::c95::{c_ushort, c_int, c_uchar};
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_ll {
                    pub sll_family: c_ushort,
                    pub sll_protocol: c_ushort,
                    pub sll_ifindex: c_int,
                    pub sll_hatype: c_ushort,
                    pub sll_pkttype: c_uchar,
                    pub sll_halen: c_uchar,
                    pub sll_addr: [c_uchar; 8]
                }
            }

        }

        #[cfg(any(target_arch = "x86_64",
                  target_arch = "aarch64"))]
        pub mod arch {
            pub mod c95 {
                #[cfg(not(target_arch = "aarch64"))]
                pub type c_char = i8;
                #[cfg(target_arch = "aarch64")]
                pub type c_char = u8;
                pub type c_schar = i8;
                pub type c_uchar = u8;
                pub type c_short = i16;
                pub type c_ushort = u16;
                pub type c_int = i32;
                pub type c_uint = u32;
                pub type c_long = i64;
                pub type c_ulong = u64;
                pub type c_float = f32;
                pub type c_double = f64;
                pub type size_t = u64;
                pub type ptrdiff_t = i64;
                pub type clock_t = i64;
                pub type time_t = i64;
                pub type suseconds_t = i64;
                #[cfg(not(target_arch = "aarch64"))]
                pub type wchar_t = i32;
                #[cfg(target_arch = "aarch64")]
                pub type wchar_t = u32;
            }
            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;
                pub type intptr_t = i64;
                pub type uintptr_t = u64;
                pub type intmax_t = i64;
                pub type uintmax_t = u64;
            }
            pub mod posix88 {
                pub type off_t = i64;
                pub type dev_t = u64;
                pub type ino_t = u64;
                pub type pid_t = i32;
                pub type uid_t = u32;
                pub type gid_t = u32;
                pub type useconds_t = u32;
                pub type mode_t = u32;
                pub type ssize_t = i64;
            }
            #[cfg(not(target_arch = "aarch64"))]
            pub mod posix01 {
                use types::os::arch::c95::{c_int, c_long, time_t};
                use types::os::arch::posix88::{dev_t, gid_t, ino_t};
                use types::os::arch::posix88::{mode_t, off_t};
                use types::os::arch::posix88::{uid_t};

                pub type nlink_t = u64;
                pub type blksize_t = i64;
                pub type blkcnt_t = i64;

                #[repr(C)]
                #[derive(Copy)] pub struct stat {
                    pub st_dev: dev_t,
                    pub st_ino: ino_t,
                    pub st_nlink: nlink_t,
                    pub st_mode: mode_t,
                    pub st_uid: uid_t,
                    pub st_gid: gid_t,
                    pub __pad0: c_int,
                    pub st_rdev: dev_t,
                    pub st_size: off_t,
                    pub st_blksize: blksize_t,
                    pub st_blocks: blkcnt_t,
                    pub st_atime: time_t,
                    pub st_atime_nsec: c_long,
                    pub st_mtime: time_t,
                    pub st_mtime_nsec: c_long,
                    pub st_ctime: time_t,
                    pub st_ctime_nsec: c_long,
                    pub __unused: [c_long; 3],
                }

                #[repr(C)]
                #[derive(Copy)] pub struct utimbuf {
                    pub actime: time_t,
                    pub modtime: time_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct pthread_attr_t {
                    pub __size: [u64; 7]
                }
            }
            #[cfg(target_arch = "aarch64")]
            pub mod posix01 {
                use types::os::arch::c95::{c_int, c_long, time_t};
                use types::os::arch::posix88::{dev_t, gid_t, ino_t};
                use types::os::arch::posix88::{mode_t, off_t};
                use types::os::arch::posix88::{uid_t};

                pub type nlink_t = u32;
                pub type blksize_t = i32;
                pub type blkcnt_t = i64;

                #[repr(C)]
                #[derive(Copy)] pub struct stat {
                    pub st_dev: dev_t,
                    pub st_ino: ino_t,
                    pub st_mode: mode_t,
                    pub st_nlink: nlink_t,
                    pub st_uid: uid_t,
                    pub st_gid: gid_t,
                    pub st_rdev: dev_t,
                    pub __pad1: dev_t,
                    pub st_size: off_t,
                    pub st_blksize: blksize_t,
                    pub __pad2: c_int,
                    pub st_blocks: blkcnt_t,
                    pub st_atime: time_t,
                    pub st_atime_nsec: c_long,
                    pub st_mtime: time_t,
                    pub st_mtime_nsec: c_long,
                    pub st_ctime: time_t,
                    pub st_ctime_nsec: c_long,
                    pub __unused: [c_int; 2],
                }

                #[repr(C)]
                #[derive(Copy)] pub struct utimbuf {
                    pub actime: time_t,
                    pub modtime: time_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct pthread_attr_t {
                    pub __size: [u64; 8]
                }
            }
            pub mod posix08 {
            }
            pub mod bsd44 {
            }
            pub mod extra {
                use types::os::arch::c95::{c_ushort, c_int, c_uchar};
                #[derive(Copy)] pub struct sockaddr_ll {
                    pub sll_family: c_ushort,
                    pub sll_protocol: c_ushort,
                    pub sll_ifindex: c_int,
                    pub sll_hatype: c_ushort,
                    pub sll_pkttype: c_uchar,
                    pub sll_halen: c_uchar,
                    pub sll_addr: [c_uchar; 8]
                }

            }
        }
    }

    #[cfg(target_os = "freebsd")]
    pub mod os {
        pub mod common {
            pub mod posix01 {
                use types::common::c95::{c_void};
                use types::os::arch::c95::{c_char, c_int, size_t,
                                                 time_t, suseconds_t, c_long};
                use types::os::arch::c99::{uintptr_t};

                pub type pthread_t = uintptr_t;

                #[repr(C)]
                #[derive(Copy)] pub struct glob_t {
                    pub gl_pathc:  size_t,
                    pub __unused1: size_t,
                    pub gl_offs:   size_t,
                    pub __unused2: c_int,
                    pub gl_pathv:  *mut *mut c_char,

                    pub __unused3: *mut c_void,

                    pub __unused4: *mut c_void,
                    pub __unused5: *mut c_void,
                    pub __unused6: *mut c_void,
                    pub __unused7: *mut c_void,
                    pub __unused8: *mut c_void,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct timeval {
                    pub tv_sec: time_t,
                    pub tv_usec: suseconds_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct timespec {
                    pub tv_sec: time_t,
                    pub tv_nsec: c_long,
                }

                #[derive(Copy)] pub enum timezone {}

                pub type sighandler_t = size_t;
            }
            pub mod bsd44 {
                use types::common::c95::{c_void};
                use types::os::arch::c95::{c_char, c_int, c_uint};

                pub type socklen_t = u32;
                pub type sa_family_t = u8;
                pub type in_port_t = u16;
                pub type in_addr_t = u32;
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr {
                    pub sa_len: u8,
                    pub sa_family: sa_family_t,
                    pub sa_data: [u8; 14],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_storage {
                    pub ss_len: u8,
                    pub ss_family: sa_family_t,
                    pub __ss_pad1: [u8; 6],
                    pub __ss_align: i64,
                    pub __ss_pad2: [u8; 112],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_in {
                    pub sin_len: u8,
                    pub sin_family: sa_family_t,
                    pub sin_port: in_port_t,
                    pub sin_addr: in_addr,
                    pub sin_zero: [u8; 8],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct in_addr {
                    pub s_addr: in_addr_t,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_in6 {
                    pub sin6_len: u8,
                    pub sin6_family: sa_family_t,
                    pub sin6_port: in_port_t,
                    pub sin6_flowinfo: u32,
                    pub sin6_addr: in6_addr,
                    pub sin6_scope_id: u32,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct in6_addr {
                    pub s6_addr: [u16; 8]
                }
                #[repr(C)]
                #[derive(Copy)] pub struct ip_mreq {
                    pub imr_multiaddr: in_addr,
                    pub imr_interface: in_addr,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct ip6_mreq {
                    pub ipv6mr_multiaddr: in6_addr,
                    pub ipv6mr_interface: c_uint,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct addrinfo {
                    pub ai_flags: c_int,
                    pub ai_family: c_int,
                    pub ai_socktype: c_int,
                    pub ai_protocol: c_int,
                    pub ai_addrlen: socklen_t,
                    pub ai_canonname: *mut c_char,
                    pub ai_addr: *mut sockaddr,
                    pub ai_next: *mut addrinfo,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_un {
                    pub sun_len: u8,
                    pub sun_family: sa_family_t,
                    pub sun_path: [c_char; 104]
                }
                #[repr(C)]
                #[derive(Copy)] pub struct ifaddrs {
                    pub ifa_next: *mut ifaddrs,
                    pub ifa_name: *mut c_char,
                    pub ifa_flags: c_uint,
                    pub ifa_addr: *mut sockaddr,
                    pub ifa_netmask: *mut sockaddr,
                    pub ifa_dstaddr: *mut sockaddr,
                    pub ifa_data: *mut c_void
                }


            }
        }

        #[cfg(target_arch = "x86_64")]
        pub mod arch {
            pub mod c95 {
                pub type c_char = i8;
                pub type c_schar = i8;
                pub type c_uchar = u8;
                pub type c_short = i16;
                pub type c_ushort = u16;
                pub type c_int = i32;
                pub type c_uint = u32;
                pub type c_long = i64;
                pub type c_ulong = u64;
                pub type c_float = f32;
                pub type c_double = f64;
                pub type size_t = u64;
                pub type ptrdiff_t = i64;
                pub type clock_t = i32;
                pub type time_t = i64;
                pub type suseconds_t = i64;
                pub type wchar_t = i32;
            }
            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;
                pub type intptr_t = i64;
                pub type uintptr_t = u64;
                pub type intmax_t = i64;
                pub type uintmax_t = u64;
            }
            pub mod posix88 {
                pub type off_t = i64;
                pub type dev_t = u32;
                pub type ino_t = u32;
                pub type pid_t = i32;
                pub type uid_t = u32;
                pub type gid_t = u32;
                pub type useconds_t = u32;
                pub type mode_t = u16;
                pub type ssize_t = i64;
            }
            pub mod posix01 {
                use types::common::c95::{c_void};
                use types::common::c99::{uint8_t, uint32_t, int32_t};
                use types::os::arch::c95::{c_long, time_t};
                use types::os::arch::posix88::{dev_t, gid_t, ino_t};
                use types::os::arch::posix88::{mode_t, off_t};
                use types::os::arch::posix88::{uid_t};

                pub type nlink_t = u16;
                pub type blksize_t = i64;
                pub type blkcnt_t = i64;
                pub type fflags_t = u32;
                #[repr(C)]
                #[derive(Copy)] pub struct stat {
                    pub st_dev: dev_t,
                    pub st_ino: ino_t,
                    pub st_mode: mode_t,
                    pub st_nlink: nlink_t,
                    pub st_uid: uid_t,
                    pub st_gid: gid_t,
                    pub st_rdev: dev_t,
                    pub st_atime: time_t,
                    pub st_atime_nsec: c_long,
                    pub st_mtime: time_t,
                    pub st_mtime_nsec: c_long,
                    pub st_ctime: time_t,
                    pub st_ctime_nsec: c_long,
                    pub st_size: off_t,
                    pub st_blocks: blkcnt_t,
                    pub st_blksize: blksize_t,
                    pub st_flags: fflags_t,
                    pub st_gen: uint32_t,
                    pub st_lspare: int32_t,
                    pub st_birthtime: time_t,
                    pub st_birthtime_nsec: c_long,
                    pub __unused: [uint8_t; 2],
                }

                #[repr(C)]
                #[derive(Copy)] pub struct utimbuf {
                    pub actime: time_t,
                    pub modtime: time_t,
                }

                pub type pthread_attr_t = *mut c_void;
            }
            pub mod posix08 {
            }
            pub mod bsd44 {
            }
            pub mod extra {
            }
        }
    }

    #[cfg(target_os = "dragonfly")]
    pub mod os {
        pub mod common {
            pub mod posix01 {
                use types::common::c95::{c_void};
                use types::os::arch::c95::{c_char, c_int, size_t,
                                                 time_t, suseconds_t, c_long};
                use types::os::arch::c99::{uintptr_t};

                pub type pthread_t = uintptr_t;

                #[repr(C)]
                #[derive(Copy)] pub struct glob_t {
                    pub gl_pathc:  size_t,
                    pub __unused1: size_t,
                    pub gl_offs:   size_t,
                    pub __unused2: c_int,
                    pub gl_pathv:  *mut *mut c_char,

                    pub __unused3: *mut c_void,

                    pub __unused4: *mut c_void,
                    pub __unused5: *mut c_void,
                    pub __unused6: *mut c_void,
                    pub __unused7: *mut c_void,
                    pub __unused8: *mut c_void,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct timeval {
                    pub tv_sec: time_t,
                    pub tv_usec: suseconds_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct timespec {
                    pub tv_sec: time_t,
                    pub tv_nsec: c_long,
                }

                #[derive(Copy)] pub enum timezone {}

                pub type sighandler_t = size_t;
            }
            pub mod bsd44 {
                use types::common::c95::{c_void};
                use types::os::arch::c95::{c_char, c_int, c_uint};

                pub type socklen_t = u32;
                pub type sa_family_t = u8;
                pub type in_port_t = u16;
                pub type in_addr_t = u32;
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr {
                    pub sa_len: u8,
                    pub sa_family: sa_family_t,
                    pub sa_data: [u8; 14],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_storage {
                    pub ss_len: u8,
                    pub ss_family: sa_family_t,
                    pub __ss_pad1: [u8; 6],
                    pub __ss_align: i64,
                    pub __ss_pad2: [u8; 112],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_in {
                    pub sin_len: u8,
                    pub sin_family: sa_family_t,
                    pub sin_port: in_port_t,
                    pub sin_addr: in_addr,
                    pub sin_zero: [u8; 8],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct in_addr {
                    pub s_addr: in_addr_t,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_in6 {
                    pub sin6_len: u8,
                    pub sin6_family: sa_family_t,
                    pub sin6_port: in_port_t,
                    pub sin6_flowinfo: u32,
                    pub sin6_addr: in6_addr,
                    pub sin6_scope_id: u32,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct in6_addr {
                    pub s6_addr: [u16; 8]
                }
                #[repr(C)]
                #[derive(Copy)] pub struct ip_mreq {
                    pub imr_multiaddr: in_addr,
                    pub imr_interface: in_addr,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct ip6_mreq {
                    pub ipv6mr_multiaddr: in6_addr,
                    pub ipv6mr_interface: c_uint,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct addrinfo {
                    pub ai_flags: c_int,
                    pub ai_family: c_int,
                    pub ai_socktype: c_int,
                    pub ai_protocol: c_int,
                    pub ai_addrlen: socklen_t,
                    pub ai_canonname: *mut c_char,
                    pub ai_addr: *mut sockaddr,
                    pub ai_next: *mut addrinfo,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_un {
                    pub sun_len: u8,
                    pub sun_family: sa_family_t,
                    pub sun_path: [c_char; 104]
                }
                #[repr(C)]
                #[derive(Copy)] pub struct ifaddrs {
                    pub ifa_next: *mut ifaddrs,
                    pub ifa_name: *mut c_char,
                    pub ifa_flags: c_uint,
                    pub ifa_addr: *mut sockaddr,
                    pub ifa_netmask: *mut sockaddr,
                    pub ifa_dstaddr: *mut sockaddr,
                    pub ifa_data: *mut c_void
                }

            }
        }

        #[cfg(target_arch = "x86_64")]
        pub mod arch {
            pub mod c95 {
                pub type c_char = i8;
                pub type c_schar = i8;
                pub type c_uchar = u8;
                pub type c_short = i16;
                pub type c_ushort = u16;
                pub type c_int = i32;
                pub type c_uint = u32;
                pub type c_long = i64;
                pub type c_ulong = u64;
                pub type c_float = f32;
                pub type c_double = f64;
                pub type size_t = u64;
                pub type ptrdiff_t = i64;
                pub type clock_t = i32;
                pub type time_t = i64;
                pub type suseconds_t = i64;
                pub type wchar_t = i32;
            }
            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;
                pub type intptr_t = i64;
                pub type uintptr_t = u64;
                pub type intmax_t = i64;
                pub type uintmax_t = u64;
            }
            pub mod posix88 {
                pub type off_t = i64;
                pub type dev_t = u32;
                pub type ino_t = u32;
                pub type pid_t = i32;
                pub type uid_t = u32;
                pub type gid_t = u32;
                pub type useconds_t = u32;
                pub type mode_t = u16;
                pub type ssize_t = i64;
            }
            pub mod posix01 {
                use types::common::c95::{c_void};
                use types::common::c99::{uint16_t, uint32_t, int32_t, uint64_t, int64_t};
                use types::os::arch::c95::{c_long, time_t};
                use types::os::arch::posix88::{dev_t, gid_t};
                use types::os::arch::posix88::{mode_t, off_t};
                use types::os::arch::posix88::{uid_t};

                pub type nlink_t = u16;
                pub type blksize_t = uint32_t;
                pub type ino_t = uint64_t;
                pub type blkcnt_t = i64;
                pub type fflags_t = u32;

                #[repr(C)]
                #[derive(Copy)] pub struct stat {
                    pub st_ino: ino_t,
                    pub st_nlink: nlink_t,
                    pub st_dev: dev_t,
                    pub st_mode: mode_t,
                    pub st_padding1: uint16_t,
                    pub st_uid: uid_t,
                    pub st_gid: gid_t,
                    pub st_rdev: dev_t,
                    pub st_atime: time_t,
                    pub st_atime_nsec: c_long,
                    pub st_mtime: time_t,
                    pub st_mtime_nsec: c_long,
                    pub st_ctime: time_t,
                    pub st_ctime_nsec: c_long,
                    pub st_size: off_t,
                    pub st_blocks: blkcnt_t,
                    pub st_blksize: blksize_t,
                    pub st_flags: fflags_t,
                    pub st_gen: uint32_t,
                    pub st_lspare: int32_t,
                    pub st_qspare1: int64_t,
                    pub st_qspare2: int64_t,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct utimbuf {
                    pub actime: time_t,
                    pub modtime: time_t,
                }

                pub type pthread_attr_t = *mut c_void;
            }
            pub mod posix08 {
            }
            pub mod bsd44 {
            }
            pub mod extra {
            }
        }
    }

    #[cfg(target_os = "openbsd")]
    pub mod os {
        pub mod common {
            pub mod posix01 {
                use types::common::c95::{c_void};
                use types::os::arch::c95::{c_char, c_int, size_t,
                                                 time_t, suseconds_t, c_long};
                use types::os::arch::c99::{uintptr_t};

                pub type pthread_t = uintptr_t;

                #[repr(C)]
                #[derive(Copy)] pub struct glob_t {
                    pub gl_pathc:  c_int,
                    pub __unused1: c_int,
                    pub gl_offs:   c_int,
                    pub __unused2: c_int,
                    pub gl_pathv:  *mut *mut c_char,

                    pub __unused3: *mut c_void,

                    pub __unused4: *mut c_void,
                    pub __unused5: *mut c_void,
                    pub __unused6: *mut c_void,
                    pub __unused7: *mut c_void,
                    pub __unused8: *mut c_void,
                    pub __unused9: *mut c_void,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct timeval {
                    pub tv_sec: time_t,
                    pub tv_usec: suseconds_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct timespec {
                    pub tv_sec: time_t,
                    pub tv_nsec: c_long,
                }

                #[derive(Copy)] pub enum timezone {}

                pub type sighandler_t = size_t;
            }
            pub mod bsd44 {
                use types::common::c95::{c_void};
                use types::os::arch::c95::{c_char, c_int, c_uint};

                pub type socklen_t = u32;
                pub type sa_family_t = u8;
                pub type in_port_t = u16;
                pub type in_addr_t = u32;
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr {
                    pub sa_len: u8,
                    pub sa_family: sa_family_t,
                    pub sa_data: [u8; 14],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_storage {
                    pub ss_len: u8,
                    pub ss_family: sa_family_t,
                    pub __ss_pad1: [u8; 6],
                    pub __ss_pad2: i64,
                    pub __ss_pad3: [u8; 240],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_in {
                    pub sin_len: u8,
                    pub sin_family: sa_family_t,
                    pub sin_port: in_port_t,
                    pub sin_addr: in_addr,
                    pub sin_zero: [u8; 8],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct in_addr {
                    pub s_addr: in_addr_t,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_in6 {
                    pub sin6_len: u8,
                    pub sin6_family: sa_family_t,
                    pub sin6_port: in_port_t,
                    pub sin6_flowinfo: u32,
                    pub sin6_addr: in6_addr,
                    pub sin6_scope_id: u32,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct in6_addr {
                    pub s6_addr: [u16; 8]
                }
                #[repr(C)]
                #[derive(Copy)] pub struct ip_mreq {
                    pub imr_multiaddr: in_addr,
                    pub imr_interface: in_addr,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct ip6_mreq {
                    pub ipv6mr_multiaddr: in6_addr,
                    pub ipv6mr_interface: c_uint,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct addrinfo {
                    pub ai_flags: c_int,
                    pub ai_family: c_int,
                    pub ai_socktype: c_int,
                    pub ai_protocol: c_int,
                    pub ai_addrlen: socklen_t,
                    pub ai_addr: *mut sockaddr,
                    pub ai_canonname: *mut c_char,
                    pub ai_next: *mut addrinfo,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_un {
                    pub sun_len: u8,
                    pub sun_family: sa_family_t,
                    pub sun_path: [c_char; 104]
                }
                #[repr(C)]
                #[derive(Copy)] pub struct ifaddrs {
                    pub ifa_next: *mut ifaddrs,
                    pub ifa_name: *mut c_char,
                    pub ifa_flags: c_uint,
                    pub ifa_addr: *mut sockaddr,
                    pub ifa_netmask: *mut sockaddr,
                    pub ifa_dstaddr: *mut sockaddr,
                    pub ifa_data: *mut c_void
                }

            }
        }

        #[cfg(target_arch = "x86_64")]
        pub mod arch {
            pub mod c95 {
                pub type c_char = i8;
                pub type c_schar = i8;
                pub type c_uchar = u8;
                pub type c_short = i16;
                pub type c_ushort = u16;
                pub type c_int = i32;
                pub type c_uint = u32;
                pub type c_long = i64;
                pub type c_ulong = u64;
                pub type c_float = f32;
                pub type c_double = f64;
                pub type size_t = u64;
                pub type ptrdiff_t = i64;
                pub type clock_t = i64;
                pub type time_t = i64;
                pub type suseconds_t = i64;
                pub type wchar_t = i32;
            }
            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;
                pub type intptr_t = i64;
                pub type uintptr_t = u64;
                pub type intmax_t = i64;
                pub type uintmax_t = u64;
            }
            pub mod posix88 {
                pub type off_t = i64;
                pub type dev_t = u32;
                pub type ino_t = u64;
                pub type pid_t = i32;
                pub type uid_t = u32;
                pub type gid_t = u32;
                pub type useconds_t = u32;
                pub type mode_t = u32;
                pub type ssize_t = i64;
            }
            pub mod posix01 {
                use types::common::c95::{c_void};
                use types::common::c99::{uint32_t, uint64_t};
                use types::os::arch::c95::{c_long, time_t};
                use types::os::arch::posix88::{dev_t, gid_t};
                use types::os::arch::posix88::{mode_t, off_t};
                use types::os::arch::posix88::{uid_t};

                pub type nlink_t = u32;
                pub type blksize_t = uint32_t;
                pub type ino_t = uint64_t;
                pub type blkcnt_t = i64;
                pub type fflags_t = u32; // type not declared, but struct stat have u_int32_t

                #[repr(C)]
                #[derive(Copy)] pub struct stat {
                    pub st_mode: mode_t,
                    pub st_dev: dev_t,
                    pub st_ino: ino_t,
                    pub st_nlink: nlink_t,
                    pub st_uid: uid_t,
                    pub st_gid: gid_t,
                    pub st_rdev: dev_t,
                    pub st_atime: time_t,
                    pub st_atime_nsec: c_long,
                    pub st_mtime: time_t,
                    pub st_mtime_nsec: c_long,
                    pub st_ctime: time_t,
                    pub st_ctime_nsec: c_long,
                    pub st_size: off_t,
                    pub st_blocks: blkcnt_t,
                    pub st_blksize: blksize_t,
                    pub st_flags: fflags_t,
                    pub st_gen: uint32_t,
                    pub st_birthtime: time_t,
                    pub st_birthtime_nsec: c_long,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct utimbuf {
                    pub actime: time_t,
                    pub modtime: time_t,
                }

                pub type pthread_attr_t = *mut c_void;
            }
            pub mod posix08 {
            }
            pub mod bsd44 {
            }
            pub mod extra {
            }
        }
    }

    #[cfg(target_os = "windows")]
    pub mod os {
        pub mod common {
            pub mod posix01 {
                use types::os::arch::c95::{c_short, time_t, c_long};
                use types::os::arch::extra::{int64, time64_t};
                use types::os::arch::posix88::{dev_t, ino_t};

                // pub Note: this is the struct called stat64 in Windows. Not stat,
                // nor stati64.
                #[repr(C)]
                #[derive(Copy)] pub struct stat {
                    pub st_dev: dev_t,
                    pub st_ino: ino_t,
                    pub st_mode: u16,
                    pub st_nlink: c_short,
                    pub st_uid: c_short,
                    pub st_gid: c_short,
                    pub st_rdev: dev_t,
                    pub st_size: int64,
                    pub st_atime: time64_t,
                    pub st_mtime: time64_t,
                    pub st_ctime: time64_t,
                }

                // note that this is called utimbuf64 in Windows
                #[repr(C)]
                #[derive(Copy)] pub struct utimbuf {
                    pub actime: time64_t,
                    pub modtime: time64_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct timeval {
                    pub tv_sec: c_long,
                    pub tv_usec: c_long,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct timespec {
                    pub tv_sec: time_t,
                    pub tv_nsec: c_long,
                }

                #[derive(Copy)] pub enum timezone {}
            }

            pub mod bsd44 {
                use types::os::arch::c95::{c_char, c_int, c_uint, size_t};
                use types::os::arch::c99::uintptr_t;

                pub type SOCKET = uintptr_t;
                pub type socklen_t = c_int;
                pub type sa_family_t = u16;
                pub type in_port_t = u16;
                pub type in_addr_t = u32;
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr {
                    pub sa_family: sa_family_t,
                    pub sa_data: [u8; 14],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_storage {
                    pub ss_family: sa_family_t,
                    pub __ss_pad1: [u8; 6],
                    pub __ss_align: i64,
                    pub __ss_pad2: [u8; 112],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_in {
                    pub sin_family: sa_family_t,
                    pub sin_port: in_port_t,
                    pub sin_addr: in_addr,
                    pub sin_zero: [u8; 8],
                }
                #[repr(C)]
                #[derive(Copy)] pub struct in_addr {
                    pub s_addr: in_addr_t,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_in6 {
                    pub sin6_family: sa_family_t,
                    pub sin6_port: in_port_t,
                    pub sin6_flowinfo: u32,
                    pub sin6_addr: in6_addr,
                    pub sin6_scope_id: u32,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct in6_addr {
                    pub s6_addr: [u16; 8]
                }
                #[repr(C)]
                #[derive(Copy)] pub struct ip_mreq {
                    pub imr_multiaddr: in_addr,
                    pub imr_interface: in_addr,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct ip6_mreq {
                    pub ipv6mr_multiaddr: in6_addr,
                    pub ipv6mr_interface: c_uint,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct addrinfo {
                    pub ai_flags: c_int,
                    pub ai_family: c_int,
                    pub ai_socktype: c_int,
                    pub ai_protocol: c_int,
                    pub ai_addrlen: size_t,
                    pub ai_canonname: *mut c_char,
                    pub ai_addr: *mut sockaddr,
                    pub ai_next: *mut addrinfo,
                }
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_un {
                    pub sun_family: sa_family_t,
                    pub sun_path: [c_char; 108]
                }
            }
        }

        pub mod arch {
            pub mod c95 {
                pub type c_char = i8;
                pub type c_schar = i8;
                pub type c_uchar = u8;
                pub type c_short = i16;
                pub type c_ushort = u16;
                pub type c_int = i32;
                pub type c_uint = u32;
                pub type c_long = i32;
                pub type c_ulong = u32;
                pub type c_float = f32;
                pub type c_double = f64;

                #[cfg(target_arch = "x86")]
                pub type size_t = u32;
                #[cfg(target_arch = "x86_64")]
                pub type size_t = u64;

                #[cfg(target_arch = "x86")]
                pub type ptrdiff_t = i32;
                #[cfg(target_arch = "x86_64")]
                pub type ptrdiff_t = i64;

                pub type clock_t = i32;

                #[cfg(target_arch = "x86")]
                pub type time_t = i32;
                #[cfg(target_arch = "x86_64")]
                pub type time_t = i64;

                #[cfg(target_arch = "x86")]
                pub type suseconds_t = i32;
                #[cfg(target_arch = "x86_64")]
                pub type suseconds_t = i64;

                pub type wchar_t = u16;
            }

            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;

                #[cfg(target_arch = "x86")]
                pub type intptr_t = i32;
                #[cfg(target_arch = "x86_64")]
                pub type intptr_t = i64;

                #[cfg(target_arch = "x86")]
                pub type uintptr_t = u32;
                #[cfg(target_arch = "x86_64")]
                pub type uintptr_t = u64;

                pub type intmax_t = i64;
                pub type uintmax_t = u64;
            }

            pub mod posix88 {
                pub type off_t = i32;
                pub type dev_t = u32;
                pub type ino_t = u16;

                pub type pid_t = u32;

                pub type useconds_t = u32;
                pub type mode_t = u16;

                #[cfg(target_arch = "x86")]
                pub type ssize_t = i32;
                #[cfg(target_arch = "x86_64")]
                pub type ssize_t = i64;
            }

            pub mod posix01 {
            }
            pub mod posix08 {
            }
            pub mod bsd44 {
            }
            pub mod extra {
                use consts::os::extra::{MAX_PROTOCOL_CHAIN,
                                              WSAPROTOCOL_LEN};
                use types::common::c95::c_void;
                use types::os::arch::c95::{c_char, c_int, c_uint, size_t};
                use types::os::arch::c95::{c_long, c_ulong};
                use types::os::arch::c95::{wchar_t};
                use types::os::arch::c99::{c_ulonglong, c_longlong, uintptr_t};

                pub type BOOL = c_int;
                pub type BYTE = u8;
                pub type BOOLEAN = BYTE;
                pub type CCHAR = c_char;
                pub type CHAR = c_char;

                pub type DWORD = c_ulong;
                pub type DWORDLONG = c_ulonglong;

                pub type HANDLE = LPVOID;
                pub type HINSTANCE = HANDLE;
                pub type HMODULE = HINSTANCE;

                pub type LONG = c_long;
                pub type PLONG = *mut c_long;

                #[cfg(target_arch = "x86")]
                pub type LONG_PTR = c_long;
                #[cfg(target_arch = "x86_64")]
                pub type LONG_PTR = i64;

                pub type LARGE_INTEGER = c_longlong;
                pub type PLARGE_INTEGER = *mut c_longlong;

                pub type LPCWSTR = *const WCHAR;
                pub type LPCSTR = *const CHAR;

                pub type LPWSTR = *mut WCHAR;
                pub type LPSTR = *mut CHAR;

                pub type LPWCH = *mut WCHAR;
                pub type LPCH = *mut CHAR;

                #[repr(C)]
                #[derive(Copy)] pub struct SECURITY_ATTRIBUTES {
                    pub nLength: DWORD,
                    pub lpSecurityDescriptor: LPVOID,
                    pub bInheritHandle: BOOL,
                }
                pub type LPSECURITY_ATTRIBUTES = *mut SECURITY_ATTRIBUTES;

                pub type LPVOID = *mut c_void;
                pub type LPCVOID = *const c_void;
                pub type LPBYTE = *mut BYTE;
                pub type LPWORD = *mut WORD;
                pub type LPDWORD = *mut DWORD;
                pub type LPHANDLE = *mut HANDLE;

                pub type LRESULT = LONG_PTR;
                pub type PBOOL = *mut BOOL;
                pub type WCHAR = wchar_t;
                pub type WORD = u16;
                pub type SIZE_T = size_t;

                pub type time64_t = i64;
                pub type int64 = i64;

                #[repr(C)]
                #[derive(Copy)] pub struct STARTUPINFO {
                    pub cb: DWORD,
                    pub lpReserved: LPWSTR,
                    pub lpDesktop: LPWSTR,
                    pub lpTitle: LPWSTR,
                    pub dwX: DWORD,
                    pub dwY: DWORD,
                    pub dwXSize: DWORD,
                    pub dwYSize: DWORD,
                    pub dwXCountChars: DWORD,
                    pub dwYCountCharts: DWORD,
                    pub dwFillAttribute: DWORD,
                    pub dwFlags: DWORD,
                    pub wShowWindow: WORD,
                    pub cbReserved2: WORD,
                    pub lpReserved2: LPBYTE,
                    pub hStdInput: HANDLE,
                    pub hStdOutput: HANDLE,
                    pub hStdError: HANDLE,
                }
                pub type LPSTARTUPINFO = *mut STARTUPINFO;

                #[repr(C)]
                #[derive(Copy)] pub struct PROCESS_INFORMATION {
                    pub hProcess: HANDLE,
                    pub hThread: HANDLE,
                    pub dwProcessId: DWORD,
                    pub dwThreadId: DWORD,
                }
                pub type LPPROCESS_INFORMATION = *mut PROCESS_INFORMATION;

                #[repr(C)]
                #[derive(Copy)] pub struct SYSTEM_INFO {
                    pub wProcessorArchitecture: WORD,
                    pub wReserved: WORD,
                    pub dwPageSize: DWORD,
                    pub lpMinimumApplicationAddress: LPVOID,
                    pub lpMaximumApplicationAddress: LPVOID,
                    pub dwActiveProcessorMask: uintptr_t,
                    pub dwNumberOfProcessors: DWORD,
                    pub dwProcessorType: DWORD,
                    pub dwAllocationGranularity: DWORD,
                    pub wProcessorLevel: WORD,
                    pub wProcessorRevision: WORD,
                }
                pub type LPSYSTEM_INFO = *mut SYSTEM_INFO;

                #[repr(C)]
                #[derive(Copy)] pub struct MEMORY_BASIC_INFORMATION {
                    pub BaseAddress: LPVOID,
                    pub AllocationBase: LPVOID,
                    pub AllocationProtect: DWORD,
                    pub RegionSize: SIZE_T,
                    pub State: DWORD,
                    pub Protect: DWORD,
                    pub Type: DWORD,
                }
                pub type LPMEMORY_BASIC_INFORMATION = *mut MEMORY_BASIC_INFORMATION;

                #[repr(C)]
                #[derive(Copy)] pub struct OVERLAPPED {
                    pub Internal: *mut c_ulong,
                    pub InternalHigh: *mut c_ulong,
                    pub Offset: DWORD,
                    pub OffsetHigh: DWORD,
                    pub hEvent: HANDLE,
                }

                pub type LPOVERLAPPED = *mut OVERLAPPED;

                #[repr(C)]
                #[derive(Copy)] pub struct FILETIME {
                    pub dwLowDateTime: DWORD,
                    pub dwHighDateTime: DWORD,
                }

                pub type LPFILETIME = *mut FILETIME;

                #[repr(C)]
                #[derive(Copy)] pub struct GUID {
                    pub Data1: DWORD,
                    pub Data2: WORD,
                    pub Data3: WORD,
                    pub Data4: [BYTE; 8],
                }

                #[repr(C)]
                #[derive(Copy)] pub struct WSAPROTOCOLCHAIN {
                    pub ChainLen: c_int,
                    pub ChainEntries: [DWORD; MAX_PROTOCOL_CHAIN as usize],
                }

                pub type LPWSAPROTOCOLCHAIN = *mut WSAPROTOCOLCHAIN;

                #[repr(C)]
                #[derive(Copy)] pub struct WSAPROTOCOL_INFO {
                    pub dwServiceFlags1: DWORD,
                    pub dwServiceFlags2: DWORD,
                    pub dwServiceFlags3: DWORD,
                    pub dwServiceFlags4: DWORD,
                    pub dwProviderFlags: DWORD,
                    pub ProviderId: GUID,
                    pub dwCatalogEntryId: DWORD,
                    pub ProtocolChain: WSAPROTOCOLCHAIN,
                    pub iVersion: c_int,
                    pub iAddressFamily: c_int,
                    pub iMaxSockAddr: c_int,
                    pub iMinSockAddr: c_int,
                    pub iSocketType: c_int,
                    pub iProtocol: c_int,
                    pub iProtocolMaxOffset: c_int,
                    pub iNetworkByteOrder: c_int,
                    pub iSecurityScheme: c_int,
                    pub dwMessageSize: DWORD,
                    pub dwProviderReserved: DWORD,
                    pub szProtocol: [u8; WSAPROTOCOL_LEN as usize + 1],
                }

                pub type LPWSAPROTOCOL_INFO = *mut WSAPROTOCOL_INFO;

                pub type GROUP = c_uint;

                #[repr(C)]
                #[derive(Copy)] pub struct WIN32_FIND_DATAW {
                    pub dwFileAttributes: DWORD,
                    pub ftCreationTime: FILETIME,
                    pub ftLastAccessTime: FILETIME,
                    pub ftLastWriteTime: FILETIME,
                    pub nFileSizeHigh: DWORD,
                    pub nFileSizeLow: DWORD,
                    pub dwReserved0: DWORD,
                    pub dwReserved1: DWORD,
                    pub cFileName: [wchar_t; 260], // #define MAX_PATH 260
                    pub cAlternateFileName: [wchar_t; 14],
                }

                pub type LPWIN32_FIND_DATAW = *mut WIN32_FIND_DATAW;
            }
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub mod os {
        pub mod common {
            pub mod posix01 {
                use types::common::c95::c_void;
                use types::os::arch::c95::{c_char, c_int, size_t, time_t};
                use types::os::arch::c95::{suseconds_t, c_long};
                use types::os::arch::c99::{uintptr_t};

                pub type pthread_t = uintptr_t;

                #[repr(C)]
                #[derive(Copy)] pub struct glob_t {
                    pub gl_pathc:  size_t,
                    pub __unused1: c_int,
                    pub gl_offs:   size_t,
                    pub __unused2: c_int,
                    pub gl_pathv:  *mut *mut c_char,

                    pub __unused3: *mut c_void,

                    pub __unused4: *mut c_void,
                    pub __unused5: *mut c_void,
                    pub __unused6: *mut c_void,
                    pub __unused7: *mut c_void,
                    pub __unused8: *mut c_void,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct timeval {
                    pub tv_sec: time_t,
                    pub tv_usec: suseconds_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct timespec {
                    pub tv_sec: time_t,
                    pub tv_nsec: c_long,
                }

                #[derive(Copy)] pub enum timezone {}

                pub type sighandler_t = size_t;
            }

            pub mod bsd44 {
                use types::common::c95::{c_void};
                use types::os::arch::c95::{c_char, c_int, c_uint};

                pub type socklen_t = u32;
                pub type sa_family_t = u8;
                pub type in_port_t = u16;
                pub type in_addr_t = u32;
                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr {
                    pub sa_len: u8,
                    pub sa_family: sa_family_t,
                    pub sa_data: [u8; 14],
                }

                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_storage {
                    pub ss_len: u8,
                    pub ss_family: sa_family_t,
                    pub __ss_pad1: [u8; 6],
                    pub __ss_align: i64,
                    pub __ss_pad2: [u8; 112],
                }

                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_in {
                    pub sin_len: u8,
                    pub sin_family: sa_family_t,
                    pub sin_port: in_port_t,
                    pub sin_addr: in_addr,
                    pub sin_zero: [u8; 8],
                }

                #[repr(C)]
                #[derive(Copy)] pub struct in_addr {
                    pub s_addr: in_addr_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_in6 {
                    pub sin6_len: u8,
                    pub sin6_family: sa_family_t,
                    pub sin6_port: in_port_t,
                    pub sin6_flowinfo: u32,
                    pub sin6_addr: in6_addr,
                    pub sin6_scope_id: u32,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct in6_addr {
                    pub s6_addr: [u16; 8]
                }

                #[repr(C)]
                #[derive(Copy)] pub struct ip_mreq {
                    pub imr_multiaddr: in_addr,
                    pub imr_interface: in_addr,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct ip6_mreq {
                    pub ipv6mr_multiaddr: in6_addr,
                    pub ipv6mr_interface: c_uint,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct addrinfo {
                    pub ai_flags: c_int,
                    pub ai_family: c_int,
                    pub ai_socktype: c_int,
                    pub ai_protocol: c_int,
                    pub ai_addrlen: socklen_t,
                    pub ai_canonname: *mut c_char,
                    pub ai_addr: *mut sockaddr,
                    pub ai_next: *mut addrinfo,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct sockaddr_un {
                    pub sun_len: u8,
                    pub sun_family: sa_family_t,
                    pub sun_path: [c_char; 104]
                }

                #[repr(C)]
                #[derive(Copy)] pub struct ifaddrs {
                    pub ifa_next: *mut ifaddrs,
                    pub ifa_name: *mut c_char,
                    pub ifa_flags: c_uint,
                    pub ifa_addr: *mut sockaddr,
                    pub ifa_netmask: *mut sockaddr,
                    pub ifa_dstaddr: *mut sockaddr,
                    pub ifa_data: *mut c_void
                }
            }
        }

        #[cfg(any(target_arch = "arm", target_arch = "x86"))]
        pub mod arch {
            pub mod c95 {
                pub type c_char = i8;
                pub type c_schar = i8;
                pub type c_uchar = u8;
                pub type c_short = i16;
                pub type c_ushort = u16;
                pub type c_int = i32;
                pub type c_uint = u32;
                pub type c_long = i32;
                pub type c_ulong = u32;
                pub type c_float = f32;
                pub type c_double = f64;
                pub type size_t = u32;
                pub type ptrdiff_t = i32;
                pub type clock_t = c_ulong;
                pub type time_t = c_long;
                pub type suseconds_t = i32;
                pub type wchar_t = i32;
            }
            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;
                pub type intptr_t = i32;
                pub type uintptr_t = u32;
                pub type intmax_t = i64;
                pub type uintmax_t = u64;
            }
            pub mod posix88 {
                use types::os::arch::c95::c_long;

                pub type off_t = i64;
                pub type dev_t = i32;
                pub type ino_t = u64;
                pub type pid_t = i32;
                pub type uid_t = u32;
                pub type gid_t = u32;
                pub type useconds_t = u32;
                pub type mode_t = u16;
                pub type ssize_t = c_long;
            }
            pub mod posix01 {
                use types::common::c99::{int32_t, int64_t, uint32_t};
                use types::os::arch::c95::{c_char, c_long, time_t};
                use types::os::arch::posix88::{dev_t, gid_t, ino_t,
                                                     mode_t, off_t, uid_t};

                pub type nlink_t = u16;
                pub type blksize_t = i32;
                pub type blkcnt_t = i64;

                #[repr(C)]
                #[derive(Copy)] pub struct stat {
                    pub st_dev: dev_t,
                    pub st_mode: mode_t,
                    pub st_nlink: nlink_t,
                    pub st_ino: ino_t,
                    pub st_uid: uid_t,
                    pub st_gid: gid_t,
                    pub st_rdev: dev_t,
                    pub st_atime: time_t,
                    pub st_atime_nsec: c_long,
                    pub st_mtime: time_t,
                    pub st_mtime_nsec: c_long,
                    pub st_ctime: time_t,
                    pub st_ctime_nsec: c_long,
                    pub st_birthtime: time_t,
                    pub st_birthtime_nsec: c_long,
                    pub st_size: off_t,
                    pub st_blocks: blkcnt_t,
                    pub st_blksize: blksize_t,
                    pub st_flags: uint32_t,
                    pub st_gen: uint32_t,
                    pub st_lspare: int32_t,
                    pub st_qspare: [int64_t; 2],
                }

                #[repr(C)]
                #[derive(Copy)] pub struct utimbuf {
                    pub actime: time_t,
                    pub modtime: time_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct pthread_attr_t {
                    pub __sig: c_long,
                    pub __opaque: [c_char; 36]
                }
            }
            pub mod posix08 {
            }
            pub mod bsd44 {
            }
            pub mod extra {
                #[repr(C)]
                #[derive(Copy)] pub struct mach_timebase_info {
                    pub numer: u32,
                    pub denom: u32,
                }

                pub type mach_timebase_info_data_t = mach_timebase_info;
            }
        }

        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        pub mod arch {
            pub mod c95 {
                pub type c_char = i8;
                pub type c_schar = i8;
                pub type c_uchar = u8;
                pub type c_short = i16;
                pub type c_ushort = u16;
                pub type c_int = i32;
                pub type c_uint = u32;
                pub type c_long = i64;
                pub type c_ulong = u64;
                pub type c_float = f32;
                pub type c_double = f64;
                pub type size_t = u64;
                pub type ptrdiff_t = i64;
                pub type clock_t = c_ulong;
                pub type time_t = c_long;
                pub type suseconds_t = i32;
                pub type wchar_t = i32;
            }
            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;
                pub type intptr_t = i64;
                pub type uintptr_t = u64;
                pub type intmax_t = i64;
                pub type uintmax_t = u64;
            }
            pub mod posix88 {
                use types::os::arch::c95::c_long;

                pub type off_t = i64;
                pub type dev_t = i32;
                pub type ino_t = u64;
                pub type pid_t = i32;
                pub type uid_t = u32;
                pub type gid_t = u32;
                pub type useconds_t = u32;
                pub type mode_t = u16;
                pub type ssize_t = c_long;
            }
            pub mod posix01 {
                use types::common::c99::{int32_t, int64_t};
                use types::common::c99::{uint32_t};
                use types::os::arch::c95::{c_char, c_long, time_t};
                use types::os::arch::posix88::{dev_t, gid_t, ino_t};
                use types::os::arch::posix88::{mode_t, off_t, uid_t};

                pub type nlink_t = u16;
                pub type blksize_t = i32;
                pub type blkcnt_t = i64;

                #[repr(C)]
                #[derive(Copy)] pub struct stat {
                    pub st_dev: dev_t,
                    pub st_mode: mode_t,
                    pub st_nlink: nlink_t,
                    pub st_ino: ino_t,
                    pub st_uid: uid_t,
                    pub st_gid: gid_t,
                    pub st_rdev: dev_t,
                    pub st_atime: time_t,
                    pub st_atime_nsec: c_long,
                    pub st_mtime: time_t,
                    pub st_mtime_nsec: c_long,
                    pub st_ctime: time_t,
                    pub st_ctime_nsec: c_long,
                    pub st_birthtime: time_t,
                    pub st_birthtime_nsec: c_long,
                    pub st_size: off_t,
                    pub st_blocks: blkcnt_t,
                    pub st_blksize: blksize_t,
                    pub st_flags: uint32_t,
                    pub st_gen: uint32_t,
                    pub st_lspare: int32_t,
                    pub st_qspare: [int64_t; 2],
                }

                #[repr(C)]
                #[derive(Copy)] pub struct utimbuf {
                    pub actime: time_t,
                    pub modtime: time_t,
                }

                #[repr(C)]
                #[derive(Copy)] pub struct pthread_attr_t {
                    pub __sig: c_long,
                    pub __opaque: [c_char; 56]
                }
            }
            pub mod posix08 {
            }
            pub mod bsd44 {
            }
            pub mod extra {
                #[repr(C)]
                #[derive(Copy)] pub struct mach_timebase_info {
                    pub numer: u32,
                    pub denom: u32,
                }

                pub type mach_timebase_info_data_t = mach_timebase_info;
            }
        }
    }
}

pub mod consts {
    // Consts tend to vary per OS so we pull their definitions out
    // into this module.

    #[cfg(target_os = "windows")]
    pub mod os {
        pub mod c95 {
            use types::os::arch::c95::{c_int, c_uint};

            pub const EXIT_FAILURE : c_int = 1;
            pub const EXIT_SUCCESS : c_int = 0;
            pub const RAND_MAX : c_int = 32767;
            pub const EOF : c_int = -1;
            pub const SEEK_SET : c_int = 0;
            pub const SEEK_CUR : c_int = 1;
            pub const SEEK_END : c_int = 2;
            pub const _IOFBF : c_int = 0;
            pub const _IONBF : c_int = 4;
            pub const _IOLBF : c_int = 64;
            pub const BUFSIZ : c_uint = 512_u32;
            pub const FOPEN_MAX : c_uint = 20_u32;
            pub const FILENAME_MAX : c_uint = 260_u32;
            pub const L_tmpnam : c_uint = 16_u32;
            pub const TMP_MAX : c_uint = 32767_u32;

            pub const WSAEINTR: c_int = 10004;
            pub const WSAEBADF: c_int = 10009;
            pub const WSAEACCES: c_int = 10013;
            pub const WSAEFAULT: c_int = 10014;
            pub const WSAEINVAL: c_int = 10022;
            pub const WSAEMFILE: c_int = 10024;
            pub const WSAEWOULDBLOCK: c_int = 10035;
            pub const WSAEINPROGRESS: c_int = 10036;
            pub const WSAEALREADY: c_int = 10037;
            pub const WSAENOTSOCK: c_int = 10038;
            pub const WSAEDESTADDRREQ: c_int = 10039;
            pub const WSAEMSGSIZE: c_int = 10040;
            pub const WSAEPROTOTYPE: c_int = 10041;
            pub const WSAENOPROTOOPT: c_int = 10042;
            pub const WSAEPROTONOSUPPORT: c_int = 10043;
            pub const WSAESOCKTNOSUPPORT: c_int = 10044;
            pub const WSAEOPNOTSUPP: c_int = 10045;
            pub const WSAEPFNOSUPPORT: c_int = 10046;
            pub const WSAEAFNOSUPPORT: c_int = 10047;
            pub const WSAEADDRINUSE: c_int = 10048;
            pub const WSAEADDRNOTAVAIL: c_int = 10049;
            pub const WSAENETDOWN: c_int = 10050;
            pub const WSAENETUNREACH: c_int = 10051;
            pub const WSAENETRESET: c_int = 10052;
            pub const WSAECONNABORTED: c_int = 10053;
            pub const WSAECONNRESET: c_int = 10054;
            pub const WSAENOBUFS: c_int = 10055;
            pub const WSAEISCONN: c_int = 10056;
            pub const WSAENOTCONN: c_int = 10057;
            pub const WSAESHUTDOWN: c_int = 10058;
            pub const WSAETOOMANYREFS: c_int = 10059;
            pub const WSAETIMEDOUT: c_int = 10060;
            pub const WSAECONNREFUSED: c_int = 10061;
            pub const WSAELOOP: c_int = 10062;
            pub const WSAENAMETOOLONG: c_int = 10063;
            pub const WSAEHOSTDOWN: c_int = 10064;
            pub const WSAEHOSTUNREACH: c_int = 10065;
            pub const WSAENOTEMPTY: c_int = 10066;
            pub const WSAEPROCLIM: c_int = 10067;
            pub const WSAEUSERS: c_int = 10068;
            pub const WSAEDQUOT: c_int = 10069;
            pub const WSAESTALE: c_int = 10070;
            pub const WSAEREMOTE: c_int = 10071;
            pub const WSASYSNOTREADY: c_int = 10091;
            pub const WSAVERNOTSUPPORTED: c_int = 10092;
            pub const WSANOTINITIALISED: c_int = 10093;
            pub const WSAEDISCON: c_int = 10101;
            pub const WSAENOMORE: c_int = 10102;
            pub const WSAECANCELLED: c_int = 10103;
            pub const WSAEINVALIDPROCTABLE: c_int = 10104;
            pub const WSAEINVALIDPROVIDER: c_int = 10105;
            pub const WSAEPROVIDERFAILEDINIT: c_int = 10106;
        }
        pub mod c99 {
        }
        pub mod posix88 {
            use types::os::arch::c95::c_int;

            pub const O_RDONLY : c_int = 0;
            pub const O_WRONLY : c_int = 1;
            pub const O_RDWR : c_int = 2;
            pub const O_APPEND : c_int = 8;
            pub const O_CREAT : c_int = 256;
            pub const O_EXCL : c_int = 1024;
            pub const O_TRUNC : c_int = 512;
            pub const S_IFIFO : c_int = 4096;
            pub const S_IFCHR : c_int = 8192;
            pub const S_IFBLK : c_int = 12288;
            pub const S_IFDIR : c_int = 16384;
            pub const S_IFREG : c_int = 32768;
            pub const S_IFLNK : c_int = 40960;
            pub const S_IFMT : c_int = 61440;
            pub const S_IEXEC : c_int = 64;
            pub const S_IWRITE : c_int = 128;
            pub const S_IREAD : c_int = 256;
            pub const S_IRWXU : c_int = 448;
            pub const S_IXUSR : c_int = 64;
            pub const S_IWUSR : c_int = 128;
            pub const S_IRUSR : c_int = 256;
            pub const F_OK : c_int = 0;
            pub const R_OK : c_int = 4;
            pub const W_OK : c_int = 2;
            pub const X_OK : c_int = 1;
            pub const STDIN_FILENO : c_int = 0;
            pub const STDOUT_FILENO : c_int = 1;
            pub const STDERR_FILENO : c_int = 2;
        }
        pub mod posix01 {
        }
        pub mod posix08 {
        }
        pub mod bsd44 {
            use types::os::arch::c95::c_int;

            pub const AF_INET: c_int = 2;
            pub const AF_INET6: c_int = 23;
            pub const SOCK_STREAM: c_int = 1;
            pub const SOCK_DGRAM: c_int = 2;
            pub const SOCK_RAW: c_int = 3;
            pub const IPPROTO_TCP: c_int = 6;
            pub const IPPROTO_IP: c_int = 0;
            pub const IPPROTO_IPV6: c_int = 41;
            pub const IP_MULTICAST_TTL: c_int = 10;
            pub const IP_MULTICAST_LOOP: c_int = 11;
            pub const IP_ADD_MEMBERSHIP: c_int = 12;
            pub const IP_DROP_MEMBERSHIP: c_int = 13;
            pub const IPV6_ADD_MEMBERSHIP: c_int = 5;
            pub const IPV6_DROP_MEMBERSHIP: c_int = 6;
            pub const IP_TTL: c_int = 4;
            pub const IP_HDRINCL: c_int = 2;

            pub const TCP_NODELAY: c_int = 0x0001;
            pub const SOL_SOCKET: c_int = 0xffff;
            pub const SO_KEEPALIVE: c_int = 8;
            pub const SO_BROADCAST: c_int = 32;
            pub const SO_REUSEADDR: c_int = 4;
            pub const SO_ERROR: c_int = 0x1007;

            pub const IFF_LOOPBACK: c_int = 4;

            pub const SHUT_RD: c_int = 0;
            pub const SHUT_WR: c_int = 1;
            pub const SHUT_RDWR: c_int = 2;
        }
        pub mod extra {
            use SOCKET;
            use types::os::arch::c95::{c_int, c_long};
            use types::os::arch::extra::{WORD, DWORD, BOOL, HANDLE};

            pub const TRUE : BOOL = 1;
            pub const FALSE : BOOL = 0;

            pub const O_TEXT : c_int = 16384;
            pub const O_BINARY : c_int = 32768;
            pub const O_NOINHERIT: c_int = 128;

            pub const ERROR_SUCCESS : c_int = 0;
            pub const ERROR_INVALID_FUNCTION: c_int = 1;
            pub const ERROR_FILE_NOT_FOUND: c_int = 2;
            pub const ERROR_ACCESS_DENIED: c_int = 5;
            pub const ERROR_INVALID_HANDLE : c_int = 6;
            pub const ERROR_BROKEN_PIPE: c_int = 109;
            pub const ERROR_DISK_FULL : c_int = 112;
            pub const ERROR_CALL_NOT_IMPLEMENTED : c_int = 120;
            pub const ERROR_INSUFFICIENT_BUFFER : c_int = 122;
            pub const ERROR_INVALID_NAME : c_int = 123;
            pub const ERROR_ALREADY_EXISTS : c_int = 183;
            pub const ERROR_PIPE_BUSY: c_int = 231;
            pub const ERROR_NO_DATA: c_int = 232;
            pub const ERROR_INVALID_ADDRESS : c_int = 487;
            pub const ERROR_PIPE_CONNECTED: c_int = 535;
            pub const ERROR_NOTHING_TO_TERMINATE: c_int = 758;
            pub const ERROR_OPERATION_ABORTED: c_int = 995;
            pub const ERROR_IO_PENDING: c_int = 997;
            pub const ERROR_FILE_INVALID : c_int = 1006;
            pub const ERROR_NOT_FOUND: c_int = 1168;
            pub const INVALID_HANDLE_VALUE: HANDLE = -1 as HANDLE;

            pub const DELETE : DWORD = 0x00010000;
            pub const READ_CONTROL : DWORD = 0x00020000;
            pub const SYNCHRONIZE : DWORD = 0x00100000;
            pub const WRITE_DAC : DWORD = 0x00040000;
            pub const WRITE_OWNER : DWORD = 0x00080000;

            pub const PROCESS_CREATE_PROCESS : DWORD = 0x0080;
            pub const PROCESS_CREATE_THREAD : DWORD = 0x0002;
            pub const PROCESS_DUP_HANDLE : DWORD = 0x0040;
            pub const PROCESS_QUERY_INFORMATION : DWORD = 0x0400;
            pub const PROCESS_QUERY_LIMITED_INFORMATION : DWORD = 0x1000;
            pub const PROCESS_SET_INFORMATION : DWORD = 0x0200;
            pub const PROCESS_SET_QUOTA : DWORD = 0x0100;
            pub const PROCESS_SUSPEND_RESUME : DWORD = 0x0800;
            pub const PROCESS_TERMINATE : DWORD = 0x0001;
            pub const PROCESS_VM_OPERATION : DWORD = 0x0008;
            pub const PROCESS_VM_READ : DWORD = 0x0010;
            pub const PROCESS_VM_WRITE : DWORD = 0x0020;

            pub const STARTF_FORCEONFEEDBACK : DWORD = 0x00000040;
            pub const STARTF_FORCEOFFFEEDBACK : DWORD = 0x00000080;
            pub const STARTF_PREVENTPINNING : DWORD = 0x00002000;
            pub const STARTF_RUNFULLSCREEN : DWORD = 0x00000020;
            pub const STARTF_TITLEISAPPID : DWORD = 0x00001000;
            pub const STARTF_TITLEISLINKNAME : DWORD = 0x00000800;
            pub const STARTF_USECOUNTCHARS : DWORD = 0x00000008;
            pub const STARTF_USEFILLATTRIBUTE : DWORD = 0x00000010;
            pub const STARTF_USEHOTKEY : DWORD = 0x00000200;
            pub const STARTF_USEPOSITION : DWORD = 0x00000004;
            pub const STARTF_USESHOWWINDOW : DWORD = 0x00000001;
            pub const STARTF_USESIZE : DWORD = 0x00000002;
            pub const STARTF_USESTDHANDLES : DWORD = 0x00000100;

            pub const WAIT_ABANDONED : DWORD = 0x00000080;
            pub const WAIT_OBJECT_0 : DWORD = 0x00000000;
            pub const WAIT_TIMEOUT : DWORD = 0x00000102;
            pub const WAIT_FAILED : DWORD = -1;

            pub const DUPLICATE_CLOSE_SOURCE : DWORD = 0x00000001;
            pub const DUPLICATE_SAME_ACCESS : DWORD = 0x00000002;

            pub const INFINITE : DWORD = -1;
            pub const STILL_ACTIVE : DWORD = 259;

            pub const MEM_COMMIT : DWORD = 0x00001000;
            pub const MEM_RESERVE : DWORD = 0x00002000;
            pub const MEM_DECOMMIT : DWORD = 0x00004000;
            pub const MEM_RELEASE : DWORD = 0x00008000;
            pub const MEM_RESET : DWORD = 0x00080000;
            pub const MEM_RESET_UNDO : DWORD = 0x1000000;
            pub const MEM_LARGE_PAGES : DWORD = 0x20000000;
            pub const MEM_PHYSICAL : DWORD = 0x00400000;
            pub const MEM_TOP_DOWN : DWORD = 0x00100000;
            pub const MEM_WRITE_WATCH : DWORD = 0x00200000;

            pub const PAGE_EXECUTE : DWORD = 0x10;
            pub const PAGE_EXECUTE_READ : DWORD = 0x20;
            pub const PAGE_EXECUTE_READWRITE : DWORD = 0x40;
            pub const PAGE_EXECUTE_WRITECOPY : DWORD = 0x80;
            pub const PAGE_NOACCESS : DWORD = 0x01;
            pub const PAGE_READONLY : DWORD = 0x02;
            pub const PAGE_READWRITE : DWORD = 0x04;
            pub const PAGE_WRITECOPY : DWORD = 0x08;
            pub const PAGE_GUARD : DWORD = 0x100;
            pub const PAGE_NOCACHE : DWORD = 0x200;
            pub const PAGE_WRITECOMBINE : DWORD = 0x400;

            pub const SEC_COMMIT : DWORD = 0x8000000;
            pub const SEC_IMAGE : DWORD = 0x1000000;
            pub const SEC_IMAGE_NO_EXECUTE : DWORD = 0x11000000;
            pub const SEC_LARGE_PAGES : DWORD = 0x80000000;
            pub const SEC_NOCACHE : DWORD = 0x10000000;
            pub const SEC_RESERVE : DWORD = 0x4000000;
            pub const SEC_WRITECOMBINE : DWORD = 0x40000000;

            pub const FILE_MAP_ALL_ACCESS : DWORD = 0xf001f;
            pub const FILE_MAP_READ : DWORD = 0x4;
            pub const FILE_MAP_WRITE : DWORD = 0x2;
            pub const FILE_MAP_COPY : DWORD = 0x1;
            pub const FILE_MAP_EXECUTE : DWORD = 0x20;

            pub const PROCESSOR_ARCHITECTURE_INTEL : WORD = 0;
            pub const PROCESSOR_ARCHITECTURE_ARM : WORD = 5;
            pub const PROCESSOR_ARCHITECTURE_IA64 : WORD = 6;
            pub const PROCESSOR_ARCHITECTURE_AMD64 : WORD = 9;
            pub const PROCESSOR_ARCHITECTURE_UNKNOWN : WORD = 0xffff;

            pub const MOVEFILE_COPY_ALLOWED: DWORD = 2;
            pub const MOVEFILE_CREATE_HARDLINK: DWORD = 16;
            pub const MOVEFILE_DELAY_UNTIL_REBOOT: DWORD = 4;
            pub const MOVEFILE_FAIL_IF_NOT_TRACKABLE: DWORD = 32;
            pub const MOVEFILE_REPLACE_EXISTING: DWORD = 1;
            pub const MOVEFILE_WRITE_THROUGH: DWORD = 8;

            pub const SYMBOLIC_LINK_FLAG_DIRECTORY: DWORD = 1;

            pub const FILE_SHARE_DELETE: DWORD = 0x4;
            pub const FILE_SHARE_READ: DWORD = 0x1;
            pub const FILE_SHARE_WRITE: DWORD = 0x2;

            pub const CREATE_ALWAYS: DWORD = 2;
            pub const CREATE_NEW: DWORD = 1;
            pub const OPEN_ALWAYS: DWORD = 4;
            pub const OPEN_EXISTING: DWORD = 3;
            pub const TRUNCATE_EXISTING: DWORD = 5;

            pub const FILE_APPEND_DATA: DWORD = 0x00000004;
            pub const FILE_READ_DATA: DWORD = 0x00000001;
            pub const FILE_WRITE_DATA: DWORD = 0x00000002;

            pub const FILE_ATTRIBUTE_ARCHIVE: DWORD = 0x20;
            pub const FILE_ATTRIBUTE_COMPRESSED: DWORD = 0x800;
            pub const FILE_ATTRIBUTE_DEVICE: DWORD = 0x40;
            pub const FILE_ATTRIBUTE_DIRECTORY: DWORD = 0x10;
            pub const FILE_ATTRIBUTE_ENCRYPTED: DWORD = 0x4000;
            pub const FILE_ATTRIBUTE_HIDDEN: DWORD = 0x2;
            pub const FILE_ATTRIBUTE_INTEGRITY_STREAM: DWORD = 0x8000;
            pub const FILE_ATTRIBUTE_NORMAL: DWORD = 0x80;
            pub const FILE_ATTRIBUTE_NOT_CONTENT_INDEXED: DWORD = 0x2000;
            pub const FILE_ATTRIBUTE_NO_SCRUB_DATA: DWORD = 0x20000;
            pub const FILE_ATTRIBUTE_OFFLINE: DWORD = 0x1000;
            pub const FILE_ATTRIBUTE_READONLY: DWORD = 0x1;
            pub const FILE_ATTRIBUTE_REPARSE_POINT: DWORD = 0x400;
            pub const FILE_ATTRIBUTE_SPARSE_FILE: DWORD = 0x200;
            pub const FILE_ATTRIBUTE_SYSTEM: DWORD = 0x4;
            pub const FILE_ATTRIBUTE_TEMPORARY: DWORD = 0x100;
            pub const FILE_ATTRIBUTE_VIRTUAL: DWORD = 0x10000;

            pub const FILE_FLAG_BACKUP_SEMANTICS: DWORD = 0x02000000;
            pub const FILE_FLAG_DELETE_ON_CLOSE: DWORD = 0x04000000;
            pub const FILE_FLAG_NO_BUFFERING: DWORD = 0x20000000;
            pub const FILE_FLAG_OPEN_NO_RECALL: DWORD = 0x00100000;
            pub const FILE_FLAG_OPEN_REPARSE_POINT: DWORD = 0x00200000;
            pub const FILE_FLAG_OVERLAPPED: DWORD = 0x40000000;
            pub const FILE_FLAG_POSIX_SEMANTICS: DWORD = 0x0100000;
            pub const FILE_FLAG_RANDOM_ACCESS: DWORD = 0x10000000;
            pub const FILE_FLAG_SESSION_AWARE: DWORD = 0x00800000;
            pub const FILE_FLAG_SEQUENTIAL_SCAN: DWORD = 0x08000000;
            pub const FILE_FLAG_WRITE_THROUGH: DWORD = 0x80000000;
            pub const FILE_FLAG_FIRST_PIPE_INSTANCE: DWORD = 0x00080000;

            pub const FILE_NAME_NORMALIZED: DWORD = 0x0;
            pub const FILE_NAME_OPENED: DWORD = 0x8;

            pub const VOLUME_NAME_DOS: DWORD = 0x0;
            pub const VOLUME_NAME_GUID: DWORD = 0x1;
            pub const VOLUME_NAME_NONE: DWORD = 0x4;
            pub const VOLUME_NAME_NT: DWORD = 0x2;

            pub const GENERIC_READ: DWORD = 0x80000000;
            pub const GENERIC_WRITE: DWORD = 0x40000000;
            pub const GENERIC_EXECUTE: DWORD = 0x20000000;
            pub const GENERIC_ALL: DWORD = 0x10000000;
            pub const FILE_WRITE_ATTRIBUTES: DWORD = 0x00000100;
            pub const FILE_READ_ATTRIBUTES: DWORD = 0x00000080;

            pub const STANDARD_RIGHTS_READ: DWORD = 0x20000;
            pub const STANDARD_RIGHTS_WRITE: DWORD = 0x20000;
            pub const FILE_WRITE_EA: DWORD = 0x00000010;
            pub const FILE_READ_EA: DWORD = 0x00000008;
            pub const FILE_GENERIC_READ: DWORD =
                STANDARD_RIGHTS_READ | FILE_READ_DATA |
                FILE_READ_ATTRIBUTES | FILE_READ_EA | SYNCHRONIZE;
            pub const FILE_GENERIC_WRITE: DWORD =
                STANDARD_RIGHTS_WRITE | FILE_WRITE_DATA |
                FILE_WRITE_ATTRIBUTES | FILE_WRITE_EA | FILE_APPEND_DATA |
                SYNCHRONIZE;

            pub const FILE_BEGIN: DWORD = 0;
            pub const FILE_CURRENT: DWORD = 1;
            pub const FILE_END: DWORD = 2;

            pub const MAX_PROTOCOL_CHAIN: DWORD = 7;
            pub const WSAPROTOCOL_LEN: DWORD = 255;
            pub const INVALID_SOCKET: SOCKET = !0;

            pub const DETACHED_PROCESS: DWORD = 0x00000008;
            pub const CREATE_NEW_PROCESS_GROUP: DWORD = 0x00000200;
            pub const CREATE_UNICODE_ENVIRONMENT: DWORD = 0x00000400;

            pub const PIPE_ACCESS_DUPLEX: DWORD = 0x00000003;
            pub const PIPE_ACCESS_INBOUND: DWORD = 0x00000001;
            pub const PIPE_ACCESS_OUTBOUND: DWORD = 0x00000002;
            pub const PIPE_TYPE_BYTE: DWORD = 0x00000000;
            pub const PIPE_TYPE_MESSAGE: DWORD = 0x00000004;
            pub const PIPE_READMODE_BYTE: DWORD = 0x00000000;
            pub const PIPE_READMODE_MESSAGE: DWORD = 0x00000002;
            pub const PIPE_WAIT: DWORD = 0x00000000;
            pub const PIPE_NOWAIT: DWORD = 0x00000001;
            pub const PIPE_ACCEPT_REMOTE_CLIENTS: DWORD = 0x00000000;
            pub const PIPE_REJECT_REMOTE_CLIENTS: DWORD = 0x00000008;
            pub const PIPE_UNLIMITED_INSTANCES: DWORD = 255;

            pub const IPPROTO_RAW: c_int = 255;

            pub const FIONBIO: c_long = -0x7FFB9982;
        }
        pub mod sysconf {
        }
    }


    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub mod os {
        pub mod c95 {
            use types::os::arch::c95::{c_int, c_uint};

            pub const EXIT_FAILURE : c_int = 1;
            pub const EXIT_SUCCESS : c_int = 0;
            pub const RAND_MAX : c_int = 2147483647;
            pub const EOF : c_int = -1;
            pub const SEEK_SET : c_int = 0;
            pub const SEEK_CUR : c_int = 1;
            pub const SEEK_END : c_int = 2;
            pub const _IOFBF : c_int = 0;
            pub const _IONBF : c_int = 2;
            pub const _IOLBF : c_int = 1;
            pub const BUFSIZ : c_uint = 8192_u32;
            pub const FOPEN_MAX : c_uint = 16_u32;
            pub const FILENAME_MAX : c_uint = 4096_u32;
            pub const L_tmpnam : c_uint = 20_u32;
            pub const TMP_MAX : c_uint = 238328_u32;
        }
        pub mod c99 {
        }
        #[cfg(any(target_arch = "x86",
                  target_arch = "x86_64",
                  target_arch = "arm",
                  target_arch = "aarch64",
                  target_arch = "powerpc"))]
        pub mod posix88 {
            use types::os::arch::c95::c_int;
            use types::common::c95::c_void;
            use types::os::arch::posix88::mode_t;

            pub const O_RDONLY : c_int = 0;
            pub const O_WRONLY : c_int = 1;
            pub const O_RDWR : c_int = 2;
            pub const O_APPEND : c_int = 1024;
            pub const O_CREAT : c_int = 64;
            pub const O_EXCL : c_int = 128;
            pub const O_TRUNC : c_int = 512;
            pub const S_IFIFO : mode_t = 4096;
            pub const S_IFCHR : mode_t = 8192;
            pub const S_IFBLK : mode_t = 24576;
            pub const S_IFDIR : mode_t = 16384;
            pub const S_IFREG : mode_t = 32768;
            pub const S_IFLNK : mode_t = 40960;
            pub const S_IFMT : mode_t = 61440;
            pub const S_IEXEC : mode_t = 64;
            pub const S_IWRITE : mode_t = 128;
            pub const S_IREAD : mode_t = 256;
            pub const S_IRWXU : mode_t = 448;
            pub const S_IXUSR : mode_t = 64;
            pub const S_IWUSR : mode_t = 128;
            pub const S_IRUSR : mode_t = 256;
            pub const F_OK : c_int = 0;
            pub const R_OK : c_int = 4;
            pub const W_OK : c_int = 2;
            pub const X_OK : c_int = 1;
            pub const STDIN_FILENO : c_int = 0;
            pub const STDOUT_FILENO : c_int = 1;
            pub const STDERR_FILENO : c_int = 2;
            pub const F_LOCK : c_int = 1;
            pub const F_TEST : c_int = 3;
            pub const F_TLOCK : c_int = 2;
            pub const F_ULOCK : c_int = 0;
            pub const SIGHUP : c_int = 1;
            pub const SIGINT : c_int = 2;
            pub const SIGQUIT : c_int = 3;
            pub const SIGILL : c_int = 4;
            pub const SIGABRT : c_int = 6;
            pub const SIGFPE : c_int = 8;
            pub const SIGKILL : c_int = 9;
            pub const SIGSEGV : c_int = 11;
            pub const SIGPIPE : c_int = 13;
            pub const SIGALRM : c_int = 14;
            pub const SIGTERM : c_int = 15;

            pub const PROT_NONE : c_int = 0;
            pub const PROT_READ : c_int = 1;
            pub const PROT_WRITE : c_int = 2;
            pub const PROT_EXEC : c_int = 4;

            pub const MAP_FILE : c_int = 0x0000;
            pub const MAP_SHARED : c_int = 0x0001;
            pub const MAP_PRIVATE : c_int = 0x0002;
            pub const MAP_FIXED : c_int = 0x0010;
            pub const MAP_ANON : c_int = 0x0020;

            pub const MAP_FAILED : *mut c_void = -1 as *mut c_void;

            pub const MCL_CURRENT : c_int = 0x0001;
            pub const MCL_FUTURE : c_int = 0x0002;

            pub const MS_ASYNC : c_int = 0x0001;
            pub const MS_INVALIDATE : c_int = 0x0002;
            pub const MS_SYNC : c_int = 0x0004;

            pub const EPERM : c_int = 1;
            pub const ENOENT : c_int = 2;
            pub const ESRCH : c_int = 3;
            pub const EINTR : c_int = 4;
            pub const EIO : c_int = 5;
            pub const ENXIO : c_int = 6;
            pub const E2BIG : c_int = 7;
            pub const ENOEXEC : c_int = 8;
            pub const EBADF : c_int = 9;
            pub const ECHILD : c_int = 10;
            pub const EAGAIN : c_int = 11;
            pub const ENOMEM : c_int = 12;
            pub const EACCES : c_int = 13;
            pub const EFAULT : c_int = 14;
            pub const ENOTBLK : c_int = 15;
            pub const EBUSY : c_int = 16;
            pub const EEXIST : c_int = 17;
            pub const EXDEV : c_int = 18;
            pub const ENODEV : c_int = 19;
            pub const ENOTDIR : c_int = 20;
            pub const EISDIR : c_int = 21;
            pub const EINVAL : c_int = 22;
            pub const ENFILE : c_int = 23;
            pub const EMFILE : c_int = 24;
            pub const ENOTTY : c_int = 25;
            pub const ETXTBSY : c_int = 26;
            pub const EFBIG : c_int = 27;
            pub const ENOSPC : c_int = 28;
            pub const ESPIPE : c_int = 29;
            pub const EROFS : c_int = 30;
            pub const EMLINK : c_int = 31;
            pub const EPIPE : c_int = 32;
            pub const EDOM : c_int = 33;
            pub const ERANGE : c_int = 34;

            pub const EDEADLK: c_int = 35;
            pub const ENAMETOOLONG: c_int = 36;
            pub const ENOLCK: c_int = 37;
            pub const ENOSYS: c_int = 38;
            pub const ENOTEMPTY: c_int = 39;
            pub const ELOOP: c_int = 40;
            pub const EWOULDBLOCK: c_int = EAGAIN;
            pub const ENOMSG: c_int = 42;
            pub const EIDRM: c_int = 43;
            pub const ECHRNG: c_int = 44;
            pub const EL2NSYNC: c_int = 45;
            pub const EL3HLT: c_int = 46;
            pub const EL3RST: c_int = 47;
            pub const ELNRNG: c_int = 48;
            pub const EUNATCH: c_int = 49;
            pub const ENOCSI: c_int = 50;
            pub const EL2HLT: c_int = 51;
            pub const EBADE: c_int = 52;
            pub const EBADR: c_int = 53;
            pub const EXFULL: c_int = 54;
            pub const ENOANO: c_int = 55;
            pub const EBADRQC: c_int = 56;
            pub const EBADSLT: c_int = 57;

            pub const EDEADLOCK: c_int = EDEADLK;

            pub const EBFONT: c_int = 59;
            pub const ENOSTR: c_int = 60;
            pub const ENODATA: c_int = 61;
            pub const ETIME: c_int = 62;
            pub const ENOSR: c_int = 63;
            pub const ENONET: c_int = 64;
            pub const ENOPKG: c_int = 65;
            pub const EREMOTE: c_int = 66;
            pub const ENOLINK: c_int = 67;
            pub const EADV: c_int = 68;
            pub const ESRMNT: c_int = 69;
            pub const ECOMM: c_int = 70;
            pub const EPROTO: c_int = 71;
            pub const EMULTIHOP: c_int = 72;
            pub const EDOTDOT: c_int = 73;
            pub const EBADMSG: c_int = 74;
            pub const EOVERFLOW: c_int = 75;
            pub const ENOTUNIQ: c_int = 76;
            pub const EBADFD: c_int = 77;
            pub const EREMCHG: c_int = 78;
            pub const ELIBACC: c_int = 79;
            pub const ELIBBAD: c_int = 80;
            pub const ELIBSCN: c_int = 81;
            pub const ELIBMAX: c_int = 82;
            pub const ELIBEXEC: c_int = 83;
            pub const EILSEQ: c_int = 84;
            pub const ERESTART: c_int = 85;
            pub const ESTRPIPE: c_int = 86;
            pub const EUSERS: c_int = 87;
            pub const ENOTSOCK: c_int = 88;
            pub const EDESTADDRREQ: c_int = 89;
            pub const EMSGSIZE: c_int = 90;
            pub const EPROTOTYPE: c_int = 91;
            pub const ENOPROTOOPT: c_int = 92;
            pub const EPROTONOSUPPORT: c_int = 93;
            pub const ESOCKTNOSUPPORT: c_int = 94;
            pub const EOPNOTSUPP: c_int = 95;
            pub const EPFNOSUPPORT: c_int = 96;
            pub const EAFNOSUPPORT: c_int = 97;
            pub const EADDRINUSE: c_int = 98;
            pub const EADDRNOTAVAIL: c_int = 99;
            pub const ENETDOWN: c_int = 100;
            pub const ENETUNREACH: c_int = 101;
            pub const ENETRESET: c_int = 102;
            pub const ECONNABORTED: c_int = 103;
            pub const ECONNRESET: c_int = 104;
            pub const ENOBUFS: c_int = 105;
            pub const EISCONN: c_int = 106;
            pub const ENOTCONN: c_int = 107;
            pub const ESHUTDOWN: c_int = 108;
            pub const ETOOMANYREFS: c_int = 109;
            pub const ETIMEDOUT: c_int = 110;
            pub const ECONNREFUSED: c_int = 111;
            pub const EHOSTDOWN: c_int = 112;
            pub const EHOSTUNREACH: c_int = 113;
            pub const EALREADY: c_int = 114;
            pub const EINPROGRESS: c_int = 115;
            pub const ESTALE: c_int = 116;
            pub const EUCLEAN: c_int = 117;
            pub const ENOTNAM: c_int = 118;
            pub const ENAVAIL: c_int = 119;
            pub const EISNAM: c_int = 120;
            pub const EREMOTEIO: c_int = 121;
            pub const EDQUOT: c_int = 122;

            pub const ENOMEDIUM: c_int = 123;
            pub const EMEDIUMTYPE: c_int = 124;
            pub const ECANCELED: c_int = 125;
            pub const ENOKEY: c_int = 126;
            pub const EKEYEXPIRED: c_int = 127;
            pub const EKEYREVOKED: c_int = 128;
            pub const EKEYREJECTED: c_int = 129;

            pub const EOWNERDEAD: c_int = 130;
            pub const ENOTRECOVERABLE: c_int = 131;

            pub const ERFKILL: c_int = 132;

            pub const EHWPOISON: c_int = 133;
        }

        #[cfg(any(target_arch = "mips",
                  target_arch = "mipsel"))]
        pub mod posix88 {
            use types::os::arch::c95::c_int;
            use types::common::c95::c_void;
            use types::os::arch::posix88::mode_t;

            pub const O_RDONLY : c_int = 0;
            pub const O_WRONLY : c_int = 1;
            pub const O_RDWR : c_int = 2;
            pub const O_APPEND : c_int = 8;
            pub const O_CREAT : c_int = 256;
            pub const O_EXCL : c_int = 1024;
            pub const O_TRUNC : c_int = 512;
            pub const S_IFIFO : mode_t = 4096;
            pub const S_IFCHR : mode_t = 8192;
            pub const S_IFBLK : mode_t = 24576;
            pub const S_IFDIR : mode_t = 16384;
            pub const S_IFREG : mode_t = 32768;
            pub const S_IFLNK : mode_t = 40960;
            pub const S_IFMT : mode_t = 61440;
            pub const S_IEXEC : mode_t = 64;
            pub const S_IWRITE : mode_t = 128;
            pub const S_IREAD : mode_t = 256;
            pub const S_IRWXU : mode_t = 448;
            pub const S_IXUSR : mode_t = 64;
            pub const S_IWUSR : mode_t = 128;
            pub const S_IRUSR : mode_t = 256;
            pub const F_OK : c_int = 0;
            pub const R_OK : c_int = 4;
            pub const W_OK : c_int = 2;
            pub const X_OK : c_int = 1;
            pub const STDIN_FILENO : c_int = 0;
            pub const STDOUT_FILENO : c_int = 1;
            pub const STDERR_FILENO : c_int = 2;
            pub const F_LOCK : c_int = 1;
            pub const F_TEST : c_int = 3;
            pub const F_TLOCK : c_int = 2;
            pub const F_ULOCK : c_int = 0;
            pub const SIGHUP : c_int = 1;
            pub const SIGINT : c_int = 2;
            pub const SIGQUIT : c_int = 3;
            pub const SIGILL : c_int = 4;
            pub const SIGABRT : c_int = 6;
            pub const SIGFPE : c_int = 8;
            pub const SIGKILL : c_int = 9;
            pub const SIGSEGV : c_int = 11;
            pub const SIGPIPE : c_int = 13;
            pub const SIGALRM : c_int = 14;
            pub const SIGTERM : c_int = 15;

            pub const PROT_NONE : c_int = 0;
            pub const PROT_READ : c_int = 1;
            pub const PROT_WRITE : c_int = 2;
            pub const PROT_EXEC : c_int = 4;

            pub const MAP_FILE : c_int = 0x0000;
            pub const MAP_SHARED : c_int = 0x0001;
            pub const MAP_PRIVATE : c_int = 0x0002;
            pub const MAP_FIXED : c_int = 0x0010;
            pub const MAP_ANON : c_int = 0x0800;

            pub const MAP_FAILED : *mut c_void = -1 as *mut c_void;

            pub const MCL_CURRENT : c_int = 0x0001;
            pub const MCL_FUTURE : c_int = 0x0002;

            pub const MS_ASYNC : c_int = 0x0001;
            pub const MS_INVALIDATE : c_int = 0x0002;
            pub const MS_SYNC : c_int = 0x0004;

            pub const EPERM : c_int = 1;
            pub const ENOENT : c_int = 2;
            pub const ESRCH : c_int = 3;
            pub const EINTR : c_int = 4;
            pub const EIO : c_int = 5;
            pub const ENXIO : c_int = 6;
            pub const E2BIG : c_int = 7;
            pub const ENOEXEC : c_int = 8;
            pub const EBADF : c_int = 9;
            pub const ECHILD : c_int = 10;
            pub const EAGAIN : c_int = 11;
            pub const ENOMEM : c_int = 12;
            pub const EACCES : c_int = 13;
            pub const EFAULT : c_int = 14;
            pub const ENOTBLK : c_int = 15;
            pub const EBUSY : c_int = 16;
            pub const EEXIST : c_int = 17;
            pub const EXDEV : c_int = 18;
            pub const ENODEV : c_int = 19;
            pub const ENOTDIR : c_int = 20;
            pub const EISDIR : c_int = 21;
            pub const EINVAL : c_int = 22;
            pub const ENFILE : c_int = 23;
            pub const EMFILE : c_int = 24;
            pub const ENOTTY : c_int = 25;
            pub const ETXTBSY : c_int = 26;
            pub const EFBIG : c_int = 27;
            pub const ENOSPC : c_int = 28;
            pub const ESPIPE : c_int = 29;
            pub const EROFS : c_int = 30;
            pub const EMLINK : c_int = 31;
            pub const EPIPE : c_int = 32;
            pub const EDOM : c_int = 33;
            pub const ERANGE : c_int = 34;

            pub const ENOMSG: c_int = 35;
            pub const EIDRM: c_int = 36;
            pub const ECHRNG: c_int = 37;
            pub const EL2NSYNC: c_int = 38;
            pub const EL3HLT: c_int = 39;
            pub const EL3RST: c_int = 40;
            pub const ELNRNG: c_int = 41;
            pub const EUNATCH: c_int = 42;
            pub const ENOCSI: c_int = 43;
            pub const EL2HLT: c_int = 44;
            pub const EDEADLK: c_int = 45;
            pub const ENOLCK: c_int = 46;
            pub const EBADE: c_int = 50;
            pub const EBADR: c_int = 51;
            pub const EXFULL: c_int = 52;
            pub const ENOANO: c_int = 53;
            pub const EBADRQC: c_int = 54;
            pub const EBADSLT: c_int = 55;
            pub const EDEADLOCK: c_int = 56;
            pub const EBFONT: c_int = 59;
            pub const ENOSTR: c_int = 60;
            pub const ENODATA: c_int = 61;
            pub const ETIME: c_int = 62;
            pub const ENOSR: c_int = 63;
            pub const ENONET: c_int = 64;
            pub const ENOPKG: c_int = 65;
            pub const EREMOTE: c_int = 66;
            pub const ENOLINK: c_int = 67;
            pub const EADV: c_int = 68;
            pub const ESRMNT: c_int = 69;
            pub const ECOMM: c_int = 70;
            pub const EPROTO: c_int = 71;
            pub const EDOTDOT: c_int = 73;
            pub const EMULTIHOP: c_int = 74;
            pub const EBADMSG: c_int = 77;
            pub const ENAMETOOLONG: c_int = 78;
            pub const EOVERFLOW: c_int = 79;
            pub const ENOTUNIQ: c_int = 80;
            pub const EBADFD: c_int = 81;
            pub const EREMCHG: c_int = 82;
            pub const ELIBACC: c_int = 83;
            pub const ELIBBAD: c_int = 84;
            pub const ELIBSCN: c_int = 95;
            pub const ELIBMAX: c_int = 86;
            pub const ELIBEXEC: c_int = 87;
            pub const EILSEQ: c_int = 88;
            pub const ENOSYS: c_int = 89;
            pub const ELOOP: c_int = 90;
            pub const ERESTART: c_int = 91;
            pub const ESTRPIPE: c_int = 92;
            pub const ENOTEMPTY: c_int = 93;
            pub const EUSERS: c_int = 94;
            pub const ENOTSOCK: c_int = 95;
            pub const EDESTADDRREQ: c_int = 96;
            pub const EMSGSIZE: c_int = 97;
            pub const EPROTOTYPE: c_int = 98;
            pub const ENOPROTOOPT: c_int = 99;
            pub const EPROTONOSUPPORT: c_int = 120;
            pub const ESOCKTNOSUPPORT: c_int = 121;
            pub const EOPNOTSUPP: c_int = 122;
            pub const EPFNOSUPPORT: c_int = 123;
            pub const EAFNOSUPPORT: c_int = 124;
            pub const EADDRINUSE: c_int = 125;
            pub const EADDRNOTAVAIL: c_int = 126;
            pub const ENETDOWN: c_int = 127;
            pub const ENETUNREACH: c_int = 128;
            pub const ENETRESET: c_int = 129;
            pub const ECONNABORTED: c_int = 130;
            pub const ECONNRESET: c_int = 131;
            pub const ENOBUFS: c_int = 132;
            pub const EISCONN: c_int = 133;
            pub const ENOTCONN: c_int = 134;
            pub const EUCLEAN: c_int = 135;
            pub const ENOTNAM: c_int = 137;
            pub const ENAVAIL: c_int = 138;
            pub const EISNAM: c_int = 139;
            pub const EREMOTEIO: c_int = 140;
            pub const ESHUTDOWN: c_int = 143;
            pub const ETOOMANYREFS: c_int = 144;
            pub const ETIMEDOUT: c_int = 145;
            pub const ECONNREFUSED: c_int = 146;
            pub const EHOSTDOWN: c_int = 147;
            pub const EHOSTUNREACH: c_int = 148;
            pub const EWOULDBLOCK: c_int = EAGAIN;
            pub const EALREADY: c_int = 149;
            pub const EINPROGRESS: c_int = 150;
            pub const ESTALE: c_int = 151;
            pub const ECANCELED: c_int = 158;

            pub const ENOMEDIUM: c_int = 159;
            pub const EMEDIUMTYPE: c_int = 160;
            pub const ENOKEY: c_int = 161;
            pub const EKEYEXPIRED: c_int = 162;
            pub const EKEYREVOKED: c_int = 163;
            pub const EKEYREJECTED: c_int = 164;

            pub const EOWNERDEAD: c_int = 165;
            pub const ENOTRECOVERABLE: c_int = 166;

            pub const ERFKILL: c_int = 167;

            pub const EHWPOISON: c_int = 168;

            pub const EDQUOT: c_int = 1133;
        }
        pub mod posix01 {
            use types::os::arch::c95::{c_int, size_t};

            pub const F_DUPFD : c_int = 0;
            pub const F_GETFD : c_int = 1;
            pub const F_SETFD : c_int = 2;
            pub const F_GETFL : c_int = 3;
            pub const F_SETFL : c_int = 4;

            pub const SIGTRAP : c_int = 5;
            pub const SIGPIPE: c_int = 13;
            pub const SIG_IGN: size_t = 1;

            pub const GLOB_ERR      : c_int = 1 << 0;
            pub const GLOB_MARK     : c_int = 1 << 1;
            pub const GLOB_NOSORT   : c_int = 1 << 2;
            pub const GLOB_DOOFFS   : c_int = 1 << 3;
            pub const GLOB_NOCHECK  : c_int = 1 << 4;
            pub const GLOB_APPEND   : c_int = 1 << 5;
            pub const GLOB_NOESCAPE : c_int = 1 << 6;

            pub const GLOB_NOSPACE  : c_int = 1;
            pub const GLOB_ABORTED  : c_int = 2;
            pub const GLOB_NOMATCH  : c_int = 3;

            pub const POSIX_MADV_NORMAL : c_int = 0;
            pub const POSIX_MADV_RANDOM : c_int = 1;
            pub const POSIX_MADV_SEQUENTIAL : c_int = 2;
            pub const POSIX_MADV_WILLNEED : c_int = 3;
            pub const POSIX_MADV_DONTNEED : c_int = 4;

            pub const _SC_MQ_PRIO_MAX : c_int = 28;
            pub const _SC_IOV_MAX : c_int = 60;
            pub const _SC_GETGR_R_SIZE_MAX : c_int = 69;
            pub const _SC_GETPW_R_SIZE_MAX : c_int = 70;
            pub const _SC_LOGIN_NAME_MAX : c_int = 71;
            pub const _SC_TTY_NAME_MAX : c_int = 72;
            pub const _SC_THREADS : c_int = 67;
            pub const _SC_THREAD_SAFE_FUNCTIONS : c_int = 68;
            pub const _SC_THREAD_DESTRUCTOR_ITERATIONS : c_int = 73;
            pub const _SC_THREAD_KEYS_MAX : c_int = 74;
            pub const _SC_THREAD_STACK_MIN : c_int = 75;
            pub const _SC_THREAD_THREADS_MAX : c_int = 76;
            pub const _SC_THREAD_ATTR_STACKADDR : c_int = 77;
            pub const _SC_THREAD_ATTR_STACKSIZE : c_int = 78;
            pub const _SC_THREAD_PRIORITY_SCHEDULING : c_int = 79;
            pub const _SC_THREAD_PRIO_INHERIT : c_int = 80;
            pub const _SC_THREAD_PRIO_PROTECT : c_int = 81;
            pub const _SC_THREAD_PROCESS_SHARED : c_int = 82;
            pub const _SC_ATEXIT_MAX : c_int = 87;
            pub const _SC_XOPEN_VERSION : c_int = 89;
            pub const _SC_XOPEN_XCU_VERSION : c_int = 90;
            pub const _SC_XOPEN_UNIX : c_int = 91;
            pub const _SC_XOPEN_CRYPT : c_int = 92;
            pub const _SC_XOPEN_ENH_I18N : c_int = 93;
            pub const _SC_XOPEN_SHM : c_int = 94;
            pub const _SC_XOPEN_LEGACY : c_int = 129;
            pub const _SC_XOPEN_REALTIME : c_int = 130;
            pub const _SC_XOPEN_REALTIME_THREADS : c_int = 131;

            pub const PTHREAD_CREATE_JOINABLE: c_int = 0;
            pub const PTHREAD_CREATE_DETACHED: c_int = 1;

            #[cfg(target_os = "android")]
            pub const PTHREAD_STACK_MIN: size_t = 8192;

            #[cfg(all(target_os = "linux",
                      any(target_arch = "arm",
                          target_arch = "x86",
                          target_arch = "x86_64")))]
            pub const PTHREAD_STACK_MIN: size_t = 16384;

            #[cfg(all(target_os = "linux",
                      any(target_arch = "mips",
                          target_arch = "mipsel",
                          target_arch = "aarch64",
                          target_arch = "powerpc")))]
            pub const PTHREAD_STACK_MIN: size_t = 131072;

            pub const CLOCK_REALTIME: c_int = 0;
            pub const CLOCK_MONOTONIC: c_int = 1;
        }
        pub mod posix08 {
        }
        #[cfg(any(target_arch = "arm",
                  target_arch = "aarch64",
                  target_arch = "x86",
                  target_arch = "x86_64",
                  target_arch = "powerpc"))]
        pub mod bsd44 {
            use types::os::arch::c95::c_int;

            pub const MADV_NORMAL : c_int = 0;
            pub const MADV_RANDOM : c_int = 1;
            pub const MADV_SEQUENTIAL : c_int = 2;
            pub const MADV_WILLNEED : c_int = 3;
            pub const MADV_DONTNEED : c_int = 4;
            pub const MADV_REMOVE : c_int = 9;
            pub const MADV_DONTFORK : c_int = 10;
            pub const MADV_DOFORK : c_int = 11;
            pub const MADV_MERGEABLE : c_int = 12;
            pub const MADV_UNMERGEABLE : c_int = 13;
            pub const MADV_HWPOISON : c_int = 100;

            pub const IFF_LOOPBACK: c_int = 0x8;

            pub const AF_UNIX: c_int = 1;
            pub const AF_INET: c_int = 2;
            pub const AF_INET6: c_int = 10;
            pub const SOCK_STREAM: c_int = 1;
            pub const SOCK_DGRAM: c_int = 2;
            pub const SOCK_RAW: c_int = 3;
            pub const IPPROTO_TCP: c_int = 6;
            pub const IPPROTO_IP: c_int = 0;
            pub const IPPROTO_IPV6: c_int = 41;
            pub const IP_MULTICAST_TTL: c_int = 33;
            pub const IP_MULTICAST_LOOP: c_int = 34;
            pub const IP_TTL: c_int = 2;
            pub const IP_HDRINCL: c_int = 3;
            pub const IP_ADD_MEMBERSHIP: c_int = 35;
            pub const IP_DROP_MEMBERSHIP: c_int = 36;
            pub const IPV6_ADD_MEMBERSHIP: c_int = 20;
            pub const IPV6_DROP_MEMBERSHIP: c_int = 21;

            pub const TCP_NODELAY: c_int = 1;
            pub const SOL_SOCKET: c_int = 1;
            pub const SO_KEEPALIVE: c_int = 9;
            pub const SO_BROADCAST: c_int = 6;
            pub const SO_REUSEADDR: c_int = 2;
            pub const SO_ERROR: c_int = 4;

            pub const SHUT_RD: c_int = 0;
            pub const SHUT_WR: c_int = 1;
            pub const SHUT_RDWR: c_int = 2;
        }
        #[cfg(any(target_arch = "mips",
                  target_arch = "mipsel"))]
        pub mod bsd44 {
            use types::os::arch::c95::c_int;

            pub const MADV_NORMAL : c_int = 0;
            pub const MADV_RANDOM : c_int = 1;
            pub const MADV_SEQUENTIAL : c_int = 2;
            pub const MADV_WILLNEED : c_int = 3;
            pub const MADV_DONTNEED : c_int = 4;
            pub const MADV_REMOVE : c_int = 9;
            pub const MADV_DONTFORK : c_int = 10;
            pub const MADV_DOFORK : c_int = 11;
            pub const MADV_MERGEABLE : c_int = 12;
            pub const MADV_UNMERGEABLE : c_int = 13;
            pub const MADV_HWPOISON : c_int = 100;

            pub const AF_UNIX: c_int = 1;
            pub const AF_INET: c_int = 2;
            pub const AF_INET6: c_int = 10;
            pub const SOCK_STREAM: c_int = 2;
            pub const SOCK_DGRAM: c_int = 1;
            pub const SOCK_RAW: c_int = 3;
            pub const IPPROTO_TCP: c_int = 6;
            pub const IPPROTO_IP: c_int = 0;
            pub const IPPROTO_IPV6: c_int = 41;
            pub const IP_MULTICAST_TTL: c_int = 33;
            pub const IP_MULTICAST_LOOP: c_int = 34;
            pub const IP_TTL: c_int = 2;
            pub const IP_HDRINCL: c_int = 3;
            pub const IP_ADD_MEMBERSHIP: c_int = 35;
            pub const IP_DROP_MEMBERSHIP: c_int = 36;
            pub const IPV6_ADD_MEMBERSHIP: c_int = 20;
            pub const IPV6_DROP_MEMBERSHIP: c_int = 21;

            pub const TCP_NODELAY: c_int = 1;
            pub const SOL_SOCKET: c_int = 65535;
            pub const SO_KEEPALIVE: c_int = 8;
            pub const SO_BROADCAST: c_int = 32;
            pub const SO_REUSEADDR: c_int = 4;
            pub const SO_ERROR: c_int = 4103;

            pub const SHUT_RD: c_int = 0;
            pub const SHUT_WR: c_int = 1;
            pub const SHUT_RDWR: c_int = 2;
        }
        #[cfg(any(target_arch = "x86",
                  target_arch = "x86_64",
                  target_arch = "arm",
                  target_arch = "aarch64",
                  target_arch = "powerpc"))]
        pub mod extra {
            use types::os::arch::c95::c_int;

            pub const AF_PACKET : c_int = 17;
            pub const IPPROTO_RAW : c_int = 255;

            pub const O_RSYNC : c_int = 1052672;
            pub const O_DSYNC : c_int = 4096;
            pub const O_NONBLOCK : c_int = 2048;
            pub const O_SYNC : c_int = 1052672;

            pub const PROT_GROWSDOWN : c_int = 0x010000000;
            pub const PROT_GROWSUP : c_int = 0x020000000;

            pub const MAP_TYPE : c_int = 0x000f;
            pub const MAP_ANONYMOUS : c_int = 0x0020;
            pub const MAP_32BIT : c_int = 0x0040;
            pub const MAP_GROWSDOWN : c_int = 0x0100;
            pub const MAP_DENYWRITE : c_int = 0x0800;
            pub const MAP_EXECUTABLE : c_int = 0x01000;
            pub const MAP_LOCKED : c_int = 0x02000;
            pub const MAP_NONRESERVE : c_int = 0x04000;
            pub const MAP_POPULATE : c_int = 0x08000;
            pub const MAP_NONBLOCK : c_int = 0x010000;
            pub const MAP_STACK : c_int = 0x020000;
        }
        #[cfg(any(target_arch = "mips",
                  target_arch = "mipsel"))]
        pub mod extra {
            use types::os::arch::c95::c_int;

            pub const AF_PACKET : c_int = 17;
            pub const IPPROTO_RAW : c_int = 255;

            pub const O_RSYNC : c_int = 16400;
            pub const O_DSYNC : c_int = 16;
            pub const O_NONBLOCK : c_int = 128;
            pub const O_SYNC : c_int = 16400;

            pub const PROT_GROWSDOWN : c_int = 0x01000000;
            pub const PROT_GROWSUP : c_int = 0x02000000;

            pub const MAP_TYPE : c_int = 0x000f;
            pub const MAP_ANONYMOUS : c_int = 0x0800;
            pub const MAP_GROWSDOWN : c_int = 0x01000;
            pub const MAP_DENYWRITE : c_int = 0x02000;
            pub const MAP_EXECUTABLE : c_int = 0x04000;
            pub const MAP_LOCKED : c_int = 0x08000;
            pub const MAP_NONRESERVE : c_int = 0x0400;
            pub const MAP_POPULATE : c_int = 0x010000;
            pub const MAP_NONBLOCK : c_int = 0x020000;
            pub const MAP_STACK : c_int = 0x040000;
        }
        #[cfg(target_os = "linux")]
        pub mod sysconf {
            use types::os::arch::c95::c_int;

            pub const _SC_ARG_MAX : c_int = 0;
            pub const _SC_CHILD_MAX : c_int = 1;
            pub const _SC_CLK_TCK : c_int = 2;
            pub const _SC_NGROUPS_MAX : c_int = 3;
            pub const _SC_OPEN_MAX : c_int = 4;
            pub const _SC_STREAM_MAX : c_int = 5;
            pub const _SC_TZNAME_MAX : c_int = 6;
            pub const _SC_JOB_CONTROL : c_int = 7;
            pub const _SC_SAVED_IDS : c_int = 8;
            pub const _SC_REALTIME_SIGNALS : c_int = 9;
            pub const _SC_PRIORITY_SCHEDULING : c_int = 10;
            pub const _SC_TIMERS : c_int = 11;
            pub const _SC_ASYNCHRONOUS_IO : c_int = 12;
            pub const _SC_PRIORITIZED_IO : c_int = 13;
            pub const _SC_SYNCHRONIZED_IO : c_int = 14;
            pub const _SC_FSYNC : c_int = 15;
            pub const _SC_MAPPED_FILES : c_int = 16;
            pub const _SC_MEMLOCK : c_int = 17;
            pub const _SC_MEMLOCK_RANGE : c_int = 18;
            pub const _SC_MEMORY_PROTECTION : c_int = 19;
            pub const _SC_MESSAGE_PASSING : c_int = 20;
            pub const _SC_SEMAPHORES : c_int = 21;
            pub const _SC_SHARED_MEMORY_OBJECTS : c_int = 22;
            pub const _SC_AIO_LISTIO_MAX : c_int = 23;
            pub const _SC_AIO_MAX : c_int = 24;
            pub const _SC_AIO_PRIO_DELTA_MAX : c_int = 25;
            pub const _SC_DELAYTIMER_MAX : c_int = 26;
            pub const _SC_MQ_OPEN_MAX : c_int = 27;
            pub const _SC_VERSION : c_int = 29;
            pub const _SC_PAGESIZE : c_int = 30;
            pub const _SC_RTSIG_MAX : c_int = 31;
            pub const _SC_SEM_NSEMS_MAX : c_int = 32;
            pub const _SC_SEM_VALUE_MAX : c_int = 33;
            pub const _SC_SIGQUEUE_MAX : c_int = 34;
            pub const _SC_TIMER_MAX : c_int = 35;
            pub const _SC_BC_BASE_MAX : c_int = 36;
            pub const _SC_BC_DIM_MAX : c_int = 37;
            pub const _SC_BC_SCALE_MAX : c_int = 38;
            pub const _SC_BC_STRING_MAX : c_int = 39;
            pub const _SC_COLL_WEIGHTS_MAX : c_int = 40;
            pub const _SC_EXPR_NEST_MAX : c_int = 42;
            pub const _SC_LINE_MAX : c_int = 43;
            pub const _SC_RE_DUP_MAX : c_int = 44;
            pub const _SC_2_VERSION : c_int = 46;
            pub const _SC_2_C_BIND : c_int = 47;
            pub const _SC_2_C_DEV : c_int = 48;
            pub const _SC_2_FORT_DEV : c_int = 49;
            pub const _SC_2_FORT_RUN : c_int = 50;
            pub const _SC_2_SW_DEV : c_int = 51;
            pub const _SC_2_LOCALEDEF : c_int = 52;
            pub const _SC_2_CHAR_TERM : c_int = 95;
            pub const _SC_2_C_VERSION : c_int = 96;
            pub const _SC_2_UPE : c_int = 97;
            pub const _SC_XBS5_ILP32_OFF32 : c_int = 125;
            pub const _SC_XBS5_ILP32_OFFBIG : c_int = 126;
            pub const _SC_XBS5_LPBIG_OFFBIG : c_int = 128;
        }
        #[cfg(target_os = "android")]
        pub mod sysconf {
            use types::os::arch::c95::c_int;

            pub const _SC_ARG_MAX : c_int = 0;
            pub const _SC_BC_BASE_MAX : c_int = 1;
            pub const _SC_BC_DIM_MAX : c_int = 2;
            pub const _SC_BC_SCALE_MAX : c_int = 3;
            pub const _SC_BC_STRING_MAX : c_int = 4;
            pub const _SC_CHILD_MAX : c_int = 5;
            pub const _SC_CLK_TCK : c_int = 6;
            pub const _SC_COLL_WEIGHTS_MAX : c_int = 7;
            pub const _SC_EXPR_NEST_MAX : c_int = 8;
            pub const _SC_LINE_MAX : c_int = 9;
            pub const _SC_NGROUPS_MAX : c_int = 10;
            pub const _SC_OPEN_MAX : c_int = 11;
            pub const _SC_2_C_BIND : c_int = 13;
            pub const _SC_2_C_DEV : c_int = 14;
            pub const _SC_2_C_VERSION : c_int = 15;
            pub const _SC_2_CHAR_TERM : c_int = 16;
            pub const _SC_2_FORT_DEV : c_int = 17;
            pub const _SC_2_FORT_RUN : c_int = 18;
            pub const _SC_2_LOCALEDEF : c_int = 19;
            pub const _SC_2_SW_DEV : c_int = 20;
            pub const _SC_2_UPE : c_int = 21;
            pub const _SC_2_VERSION : c_int = 22;
            pub const _SC_JOB_CONTROL : c_int = 23;
            pub const _SC_SAVED_IDS : c_int = 24;
            pub const _SC_VERSION : c_int = 25;
            pub const _SC_RE_DUP_MAX : c_int = 26;
            pub const _SC_STREAM_MAX : c_int = 27;
            pub const _SC_TZNAME_MAX : c_int = 28;
            pub const _SC_PAGESIZE : c_int = 39;
        }
    }

    #[cfg(any(target_os = "freebsd",
              target_os = "dragonfly"))]
    pub mod os {
        pub mod c95 {
            use types::os::arch::c95::{c_int, c_uint};

            pub const EXIT_FAILURE : c_int = 1;
            pub const EXIT_SUCCESS : c_int = 0;
            pub const RAND_MAX : c_int = 2147483647;
            pub const EOF : c_int = -1;
            pub const SEEK_SET : c_int = 0;
            pub const SEEK_CUR : c_int = 1;
            pub const SEEK_END : c_int = 2;
            pub const _IOFBF : c_int = 0;
            pub const _IONBF : c_int = 2;
            pub const _IOLBF : c_int = 1;
            pub const BUFSIZ : c_uint = 1024_u32;
            pub const FOPEN_MAX : c_uint = 20_u32;
            pub const FILENAME_MAX : c_uint = 1024_u32;
            pub const L_tmpnam : c_uint = 1024_u32;
            pub const TMP_MAX : c_uint = 308915776_u32;
        }
        pub mod c99 {
        }
        pub mod posix88 {
            use types::common::c95::c_void;
            use types::os::arch::c95::c_int;
            use types::os::arch::posix88::mode_t;

            pub const O_RDONLY : c_int = 0;
            pub const O_WRONLY : c_int = 1;
            pub const O_RDWR : c_int = 2;
            pub const O_APPEND : c_int = 8;
            pub const O_CREAT : c_int = 512;
            pub const O_EXCL : c_int = 2048;
            pub const O_TRUNC : c_int = 1024;
            pub const S_IFIFO : mode_t = 4096;
            pub const S_IFCHR : mode_t = 8192;
            pub const S_IFBLK : mode_t = 24576;
            pub const S_IFDIR : mode_t = 16384;
            pub const S_IFREG : mode_t = 32768;
            pub const S_IFLNK : mode_t = 40960;
            pub const S_IFMT : mode_t = 61440;
            pub const S_IEXEC : mode_t = 64;
            pub const S_IWRITE : mode_t = 128;
            pub const S_IREAD : mode_t = 256;
            pub const S_IRWXU : mode_t = 448;
            pub const S_IXUSR : mode_t = 64;
            pub const S_IWUSR : mode_t = 128;
            pub const S_IRUSR : mode_t = 256;
            pub const F_OK : c_int = 0;
            pub const R_OK : c_int = 4;
            pub const W_OK : c_int = 2;
            pub const X_OK : c_int = 1;
            pub const STDIN_FILENO : c_int = 0;
            pub const STDOUT_FILENO : c_int = 1;
            pub const STDERR_FILENO : c_int = 2;
            pub const F_LOCK : c_int = 1;
            pub const F_TEST : c_int = 3;
            pub const F_TLOCK : c_int = 2;
            pub const F_ULOCK : c_int = 0;
            pub const SIGHUP : c_int = 1;
            pub const SIGINT : c_int = 2;
            pub const SIGQUIT : c_int = 3;
            pub const SIGILL : c_int = 4;
            pub const SIGABRT : c_int = 6;
            pub const SIGFPE : c_int = 8;
            pub const SIGKILL : c_int = 9;
            pub const SIGSEGV : c_int = 11;
            pub const SIGPIPE : c_int = 13;
            pub const SIGALRM : c_int = 14;
            pub const SIGTERM : c_int = 15;

            pub const PROT_NONE : c_int = 0;
            pub const PROT_READ : c_int = 1;
            pub const PROT_WRITE : c_int = 2;
            pub const PROT_EXEC : c_int = 4;

            pub const MAP_FILE : c_int = 0x0000;
            pub const MAP_SHARED : c_int = 0x0001;
            pub const MAP_PRIVATE : c_int = 0x0002;
            pub const MAP_FIXED : c_int = 0x0010;
            pub const MAP_ANON : c_int = 0x1000;

            pub const MAP_FAILED : *mut c_void = -1 as *mut c_void;

            pub const MCL_CURRENT : c_int = 0x0001;
            pub const MCL_FUTURE : c_int = 0x0002;

            pub const MS_SYNC : c_int = 0x0000;
            pub const MS_ASYNC : c_int = 0x0001;
            pub const MS_INVALIDATE : c_int = 0x0002;

            pub const EPERM : c_int = 1;
            pub const ENOENT : c_int = 2;
            pub const ESRCH : c_int = 3;
            pub const EINTR : c_int = 4;
            pub const EIO : c_int = 5;
            pub const ENXIO : c_int = 6;
            pub const E2BIG : c_int = 7;
            pub const ENOEXEC : c_int = 8;
            pub const EBADF : c_int = 9;
            pub const ECHILD : c_int = 10;
            pub const EDEADLK : c_int = 11;
            pub const ENOMEM : c_int = 12;
            pub const EACCES : c_int = 13;
            pub const EFAULT : c_int = 14;
            pub const ENOTBLK : c_int = 15;
            pub const EBUSY : c_int = 16;
            pub const EEXIST : c_int = 17;
            pub const EXDEV : c_int = 18;
            pub const ENODEV : c_int = 19;
            pub const ENOTDIR : c_int = 20;
            pub const EISDIR : c_int = 21;
            pub const EINVAL : c_int = 22;
            pub const ENFILE : c_int = 23;
            pub const EMFILE : c_int = 24;
            pub const ENOTTY : c_int = 25;
            pub const ETXTBSY : c_int = 26;
            pub const EFBIG : c_int = 27;
            pub const ENOSPC : c_int = 28;
            pub const ESPIPE : c_int = 29;
            pub const EROFS : c_int = 30;
            pub const EMLINK : c_int = 31;
            pub const EPIPE : c_int = 32;
            pub const EDOM : c_int = 33;
            pub const ERANGE : c_int = 34;
            pub const EAGAIN : c_int = 35;
            pub const EWOULDBLOCK : c_int = 35;
            pub const EINPROGRESS : c_int = 36;
            pub const EALREADY : c_int = 37;
            pub const ENOTSOCK : c_int = 38;
            pub const EDESTADDRREQ : c_int = 39;
            pub const EMSGSIZE : c_int = 40;
            pub const EPROTOTYPE : c_int = 41;
            pub const ENOPROTOOPT : c_int = 42;
            pub const EPROTONOSUPPORT : c_int = 43;
            pub const ESOCKTNOSUPPORT : c_int = 44;
            pub const EOPNOTSUPP : c_int = 45;
            pub const EPFNOSUPPORT : c_int = 46;
            pub const EAFNOSUPPORT : c_int = 47;
            pub const EADDRINUSE : c_int = 48;
            pub const EADDRNOTAVAIL : c_int = 49;
            pub const ENETDOWN : c_int = 50;
            pub const ENETUNREACH : c_int = 51;
            pub const ENETRESET : c_int = 52;
            pub const ECONNABORTED : c_int = 53;
            pub const ECONNRESET : c_int = 54;
            pub const ENOBUFS : c_int = 55;
            pub const EISCONN : c_int = 56;
            pub const ENOTCONN : c_int = 57;
            pub const ESHUTDOWN : c_int = 58;
            pub const ETOOMANYREFS : c_int = 59;
            pub const ETIMEDOUT : c_int = 60;
            pub const ECONNREFUSED : c_int = 61;
            pub const ELOOP : c_int = 62;
            pub const ENAMETOOLONG : c_int = 63;
            pub const EHOSTDOWN : c_int = 64;
            pub const EHOSTUNREACH : c_int = 65;
            pub const ENOTEMPTY : c_int = 66;
            pub const EPROCLIM : c_int = 67;
            pub const EUSERS : c_int = 68;
            pub const EDQUOT : c_int = 69;
            pub const ESTALE : c_int = 70;
            pub const EREMOTE : c_int = 71;
            pub const EBADRPC : c_int = 72;
            pub const ERPCMISMATCH : c_int = 73;
            pub const EPROGUNAVAIL : c_int = 74;
            pub const EPROGMISMATCH : c_int = 75;
            pub const EPROCUNAVAIL : c_int = 76;
            pub const ENOLCK : c_int = 77;
            pub const ENOSYS : c_int = 78;
            pub const EFTYPE : c_int = 79;
            pub const EAUTH : c_int = 80;
            pub const ENEEDAUTH : c_int = 81;
            pub const EIDRM : c_int = 82;
            pub const ENOMSG : c_int = 83;
            pub const EOVERFLOW : c_int = 84;
            pub const ECANCELED : c_int = 85;
            pub const EILSEQ : c_int = 86;
            pub const ENOATTR : c_int = 87;
            pub const EDOOFUS : c_int = 88;
            pub const EBADMSG : c_int = 89;
            pub const EMULTIHOP : c_int = 90;
            pub const ENOLINK : c_int = 91;
            pub const EPROTO : c_int = 92;
            pub const ENOMEDIUM : c_int = 93;
            pub const EUNUSED94 : c_int = 94;
            pub const EUNUSED95 : c_int = 95;
            pub const EUNUSED96 : c_int = 96;
            pub const EUNUSED97 : c_int = 97;
            pub const EUNUSED98 : c_int = 98;
            pub const EASYNC : c_int = 99;
            pub const ELAST : c_int = 99;
        }
        pub mod posix01 {
            use types::os::arch::c95::{c_int, size_t};

            pub const F_DUPFD : c_int = 0;
            pub const F_GETFD : c_int = 1;
            pub const F_SETFD : c_int = 2;
            pub const F_GETFL : c_int = 3;
            pub const F_SETFL : c_int = 4;

            pub const SIGTRAP : c_int = 5;
            pub const SIGPIPE: c_int = 13;
            pub const SIG_IGN: size_t = 1;

            pub const GLOB_APPEND   : c_int = 0x0001;
            pub const GLOB_DOOFFS   : c_int = 0x0002;
            pub const GLOB_ERR      : c_int = 0x0004;
            pub const GLOB_MARK     : c_int = 0x0008;
            pub const GLOB_NOCHECK  : c_int = 0x0010;
            pub const GLOB_NOSORT   : c_int = 0x0020;
            pub const GLOB_NOESCAPE : c_int = 0x2000;

            pub const GLOB_NOSPACE  : c_int = -1;
            pub const GLOB_ABORTED  : c_int = -2;
            pub const GLOB_NOMATCH  : c_int = -3;

            pub const POSIX_MADV_NORMAL : c_int = 0;
            pub const POSIX_MADV_RANDOM : c_int = 1;
            pub const POSIX_MADV_SEQUENTIAL : c_int = 2;
            pub const POSIX_MADV_WILLNEED : c_int = 3;
            pub const POSIX_MADV_DONTNEED : c_int = 4;

            pub const _SC_IOV_MAX : c_int = 56;
            pub const _SC_GETGR_R_SIZE_MAX : c_int = 70;
            pub const _SC_GETPW_R_SIZE_MAX : c_int = 71;
            pub const _SC_LOGIN_NAME_MAX : c_int = 73;
            pub const _SC_MQ_PRIO_MAX : c_int = 75;
            pub const _SC_THREAD_ATTR_STACKADDR : c_int = 82;
            pub const _SC_THREAD_ATTR_STACKSIZE : c_int = 83;
            pub const _SC_THREAD_DESTRUCTOR_ITERATIONS : c_int = 85;
            pub const _SC_THREAD_KEYS_MAX : c_int = 86;
            pub const _SC_THREAD_PRIO_INHERIT : c_int = 87;
            pub const _SC_THREAD_PRIO_PROTECT : c_int = 88;
            pub const _SC_THREAD_PRIORITY_SCHEDULING : c_int = 89;
            pub const _SC_THREAD_PROCESS_SHARED : c_int = 90;
            pub const _SC_THREAD_SAFE_FUNCTIONS : c_int = 91;
            pub const _SC_THREAD_STACK_MIN : c_int = 93;
            pub const _SC_THREAD_THREADS_MAX : c_int = 94;
            pub const _SC_THREADS : c_int = 96;
            pub const _SC_TTY_NAME_MAX : c_int = 101;
            pub const _SC_ATEXIT_MAX : c_int = 107;
            pub const _SC_XOPEN_CRYPT : c_int = 108;
            pub const _SC_XOPEN_ENH_I18N : c_int = 109;
            pub const _SC_XOPEN_LEGACY : c_int = 110;
            pub const _SC_XOPEN_REALTIME : c_int = 111;
            pub const _SC_XOPEN_REALTIME_THREADS : c_int = 112;
            pub const _SC_XOPEN_SHM : c_int = 113;
            pub const _SC_XOPEN_UNIX : c_int = 115;
            pub const _SC_XOPEN_VERSION : c_int = 116;
            pub const _SC_XOPEN_XCU_VERSION : c_int = 117;

            pub const PTHREAD_CREATE_JOINABLE: c_int = 0;
            pub const PTHREAD_CREATE_DETACHED: c_int = 1;

            #[cfg(target_arch = "arm")]
            pub const PTHREAD_STACK_MIN: size_t = 4096;

            #[cfg(all(target_os = "freebsd",
                      any(target_arch = "mips",
                          target_arch = "mipsel",
                          target_arch = "x86",
                          target_arch = "x86_64")))]
            pub const PTHREAD_STACK_MIN: size_t = 2048;

            #[cfg(target_os = "dragonfly")]
            pub const PTHREAD_STACK_MIN: size_t = 1024;

            pub const CLOCK_REALTIME: c_int = 0;
            pub const CLOCK_MONOTONIC: c_int = 4;
        }
        pub mod posix08 {
        }
        pub mod bsd44 {
            use types::os::arch::c95::c_int;

            pub const MADV_NORMAL : c_int = 0;
            pub const MADV_RANDOM : c_int = 1;
            pub const MADV_SEQUENTIAL : c_int = 2;
            pub const MADV_WILLNEED : c_int = 3;
            pub const MADV_DONTNEED : c_int = 4;
            pub const MADV_FREE : c_int = 5;
            pub const MADV_NOSYNC : c_int = 6;
            pub const MADV_AUTOSYNC : c_int = 7;
            pub const MADV_NOCORE : c_int = 8;
            pub const MADV_CORE : c_int = 9;
            pub const MADV_PROTECT : c_int = 10;

            pub const MINCORE_INCORE : c_int =  0x1;
            pub const MINCORE_REFERENCED : c_int = 0x2;
            pub const MINCORE_MODIFIED : c_int = 0x4;
            pub const MINCORE_REFERENCED_OTHER : c_int = 0x8;
            pub const MINCORE_MODIFIED_OTHER : c_int = 0x10;
            pub const MINCORE_SUPER : c_int = 0x20;

            pub const AF_INET: c_int = 2;
            pub const AF_INET6: c_int = 28;
            pub const AF_UNIX: c_int = 1;
            pub const SOCK_STREAM: c_int = 1;
            pub const SOCK_DGRAM: c_int = 2;
            pub const SOCK_RAW: c_int = 3;
            pub const IPPROTO_TCP: c_int = 6;
            pub const IPPROTO_IP: c_int = 0;
            pub const IPPROTO_IPV6: c_int = 41;
            pub const IP_MULTICAST_TTL: c_int = 10;
            pub const IP_MULTICAST_LOOP: c_int = 11;
            pub const IP_TTL: c_int = 4;
            pub const IP_HDRINCL: c_int = 2;
            pub const IP_ADD_MEMBERSHIP: c_int = 12;
            pub const IP_DROP_MEMBERSHIP: c_int = 13;
            pub const IPV6_ADD_MEMBERSHIP: c_int = 12;
            pub const IPV6_DROP_MEMBERSHIP: c_int = 13;

            pub const TCP_NODELAY: c_int = 1;
            pub const TCP_KEEPIDLE: c_int = 256;
            pub const SOL_SOCKET: c_int = 0xffff;
            pub const SO_KEEPALIVE: c_int = 0x0008;
            pub const SO_BROADCAST: c_int = 0x0020;
            pub const SO_REUSEADDR: c_int = 0x0004;
            pub const SO_ERROR: c_int = 0x1007;

            pub const IFF_LOOPBACK: c_int = 0x8;

            pub const SHUT_RD: c_int = 0;
            pub const SHUT_WR: c_int = 1;
            pub const SHUT_RDWR: c_int = 2;
        }
        pub mod extra {
            use types::os::arch::c95::c_int;

            pub const O_SYNC : c_int = 128;
            pub const O_NONBLOCK : c_int = 4;
            pub const CTL_KERN: c_int = 1;
            pub const KERN_PROC: c_int = 14;
            #[cfg(target_os = "freebsd")]
            pub const KERN_PROC_PATHNAME: c_int = 12;
            #[cfg(target_os = "dragonfly")]
            pub const KERN_PROC_PATHNAME: c_int = 9;

            pub const MAP_COPY : c_int = 0x0002;
            pub const MAP_RENAME : c_int = 0x0020;
            pub const MAP_NORESERVE : c_int = 0x0040;
            pub const MAP_HASSEMAPHORE : c_int = 0x0200;
            pub const MAP_STACK : c_int = 0x0400;
            pub const MAP_NOSYNC : c_int = 0x0800;
            pub const MAP_NOCORE : c_int = 0x020000;

            pub const IPPROTO_RAW : c_int = 255;
        }
        pub mod sysconf {
            use types::os::arch::c95::c_int;

            pub const _SC_ARG_MAX : c_int = 1;
            pub const _SC_CHILD_MAX : c_int = 2;
            pub const _SC_CLK_TCK : c_int = 3;
            pub const _SC_NGROUPS_MAX : c_int = 4;
            pub const _SC_OPEN_MAX : c_int = 5;
            pub const _SC_JOB_CONTROL : c_int = 6;
            pub const _SC_SAVED_IDS : c_int = 7;
            pub const _SC_VERSION : c_int = 8;
            pub const _SC_BC_BASE_MAX : c_int = 9;
            pub const _SC_BC_DIM_MAX : c_int = 10;
            pub const _SC_BC_SCALE_MAX : c_int = 11;
            pub const _SC_BC_STRING_MAX : c_int = 12;
            pub const _SC_COLL_WEIGHTS_MAX : c_int = 13;
            pub const _SC_EXPR_NEST_MAX : c_int = 14;
            pub const _SC_LINE_MAX : c_int = 15;
            pub const _SC_RE_DUP_MAX : c_int = 16;
            pub const _SC_2_VERSION : c_int = 17;
            pub const _SC_2_C_BIND : c_int = 18;
            pub const _SC_2_C_DEV : c_int = 19;
            pub const _SC_2_CHAR_TERM : c_int = 20;
            pub const _SC_2_FORT_DEV : c_int = 21;
            pub const _SC_2_FORT_RUN : c_int = 22;
            pub const _SC_2_LOCALEDEF : c_int = 23;
            pub const _SC_2_SW_DEV : c_int = 24;
            pub const _SC_2_UPE : c_int = 25;
            pub const _SC_STREAM_MAX : c_int = 26;
            pub const _SC_TZNAME_MAX : c_int = 27;
            pub const _SC_ASYNCHRONOUS_IO : c_int = 28;
            pub const _SC_MAPPED_FILES : c_int = 29;
            pub const _SC_MEMLOCK : c_int = 30;
            pub const _SC_MEMLOCK_RANGE : c_int = 31;
            pub const _SC_MEMORY_PROTECTION : c_int = 32;
            pub const _SC_MESSAGE_PASSING : c_int = 33;
            pub const _SC_PRIORITIZED_IO : c_int = 34;
            pub const _SC_PRIORITY_SCHEDULING : c_int = 35;
            pub const _SC_REALTIME_SIGNALS : c_int = 36;
            pub const _SC_SEMAPHORES : c_int = 37;
            pub const _SC_FSYNC : c_int = 38;
            pub const _SC_SHARED_MEMORY_OBJECTS : c_int = 39;
            pub const _SC_SYNCHRONIZED_IO : c_int = 40;
            pub const _SC_TIMERS : c_int = 41;
            pub const _SC_AIO_LISTIO_MAX : c_int = 42;
            pub const _SC_AIO_MAX : c_int = 43;
            pub const _SC_AIO_PRIO_DELTA_MAX : c_int = 44;
            pub const _SC_DELAYTIMER_MAX : c_int = 45;
            pub const _SC_MQ_OPEN_MAX : c_int = 46;
            pub const _SC_PAGESIZE : c_int = 47;
            pub const _SC_RTSIG_MAX : c_int = 48;
            pub const _SC_SEM_NSEMS_MAX : c_int = 49;
            pub const _SC_SEM_VALUE_MAX : c_int = 50;
            pub const _SC_SIGQUEUE_MAX : c_int = 51;
            pub const _SC_TIMER_MAX : c_int = 52;
        }
    }

    #[cfg(target_os = "openbsd")]
    pub mod os {
        pub mod c95 {
            use types::os::arch::c95::{c_int, c_uint};

            pub const EXIT_FAILURE : c_int = 1;
            pub const EXIT_SUCCESS : c_int = 0;
            pub const RAND_MAX : c_int = 2147483647;
            pub const EOF : c_int = -1;
            pub const SEEK_SET : c_int = 0;
            pub const SEEK_CUR : c_int = 1;
            pub const SEEK_END : c_int = 2;
            pub const _IOFBF : c_int = 0;
            pub const _IONBF : c_int = 2;
            pub const _IOLBF : c_int = 1;
            pub const BUFSIZ : c_uint = 1024_u32;
            pub const FOPEN_MAX : c_uint = 20_u32;
            pub const FILENAME_MAX : c_uint = 1024_u32;
            pub const L_tmpnam : c_uint = 1024_u32;
            pub const TMP_MAX : c_uint = 308915776_u32;
        }
        pub mod c99 {
        }
        pub mod posix88 {
            use types::common::c95::c_void;
            use types::os::arch::c95::c_int;
            use types::os::arch::posix88::mode_t;

            pub const O_RDONLY : c_int = 0;
            pub const O_WRONLY : c_int = 1;
            pub const O_RDWR : c_int = 2;
            pub const O_APPEND : c_int = 8;
            pub const O_CREAT : c_int = 512;
            pub const O_EXCL : c_int = 2048;
            pub const O_TRUNC : c_int = 1024;
            pub const S_IFIFO : mode_t = 4096;
            pub const S_IFCHR : mode_t = 8192;
            pub const S_IFBLK : mode_t = 24576;
            pub const S_IFDIR : mode_t = 16384;
            pub const S_IFREG : mode_t = 32768;
            pub const S_IFLNK : mode_t = 40960;
            pub const S_IFMT : mode_t = 61440;
            pub const S_IEXEC : mode_t = 64;
            pub const S_IWRITE : mode_t = 128;
            pub const S_IREAD : mode_t = 256;
            pub const S_IRWXU : mode_t = 448;
            pub const S_IXUSR : mode_t = 64;
            pub const S_IWUSR : mode_t = 128;
            pub const S_IRUSR : mode_t = 256;
            pub const F_OK : c_int = 0;
            pub const R_OK : c_int = 4;
            pub const W_OK : c_int = 2;
            pub const X_OK : c_int = 1;
            pub const STDIN_FILENO : c_int = 0;
            pub const STDOUT_FILENO : c_int = 1;
            pub const STDERR_FILENO : c_int = 2;
            pub const F_LOCK : c_int = 1;
            pub const F_TEST : c_int = 3;
            pub const F_TLOCK : c_int = 2;
            pub const F_ULOCK : c_int = 0;
            pub const SIGHUP : c_int = 1;
            pub const SIGINT : c_int = 2;
            pub const SIGQUIT : c_int = 3;
            pub const SIGILL : c_int = 4;
            pub const SIGABRT : c_int = 6;
            pub const SIGFPE : c_int = 8;
            pub const SIGKILL : c_int = 9;
            pub const SIGSEGV : c_int = 11;
            pub const SIGPIPE : c_int = 13;
            pub const SIGALRM : c_int = 14;
            pub const SIGTERM : c_int = 15;

            pub const PROT_NONE : c_int = 0;
            pub const PROT_READ : c_int = 1;
            pub const PROT_WRITE : c_int = 2;
            pub const PROT_EXEC : c_int = 4;

            pub const MAP_FILE : c_int = 0x0000;
            pub const MAP_SHARED : c_int = 0x0001;
            pub const MAP_PRIVATE : c_int = 0x0002;
            pub const MAP_FIXED : c_int = 0x0010;
            pub const MAP_ANON : c_int = 0x1000;

            pub const MAP_FAILED : *mut c_void = -1 as *mut c_void;

            pub const MCL_CURRENT : c_int = 0x0001;
            pub const MCL_FUTURE : c_int = 0x0002;

            pub const MS_SYNC : c_int = 0x0002; // changed
            pub const MS_ASYNC : c_int = 0x0001;
            pub const MS_INVALIDATE : c_int = 0x0004; // changed

            pub const EPERM : c_int = 1; // not checked
            pub const ENOENT : c_int = 2;
            pub const ESRCH : c_int = 3;
            pub const EINTR : c_int = 4;
            pub const EIO : c_int = 5;
            pub const ENXIO : c_int = 6;
            pub const E2BIG : c_int = 7;
            pub const ENOEXEC : c_int = 8;
            pub const EBADF : c_int = 9;
            pub const ECHILD : c_int = 10;
            pub const EDEADLK : c_int = 11;
            pub const ENOMEM : c_int = 12;
            pub const EACCES : c_int = 13;
            pub const EFAULT : c_int = 14;
            pub const ENOTBLK : c_int = 15;
            pub const EBUSY : c_int = 16;
            pub const EEXIST : c_int = 17;
            pub const EXDEV : c_int = 18;
            pub const ENODEV : c_int = 19;
            pub const ENOTDIR : c_int = 20;
            pub const EISDIR : c_int = 21;
            pub const EINVAL : c_int = 22;
            pub const ENFILE : c_int = 23;
            pub const EMFILE : c_int = 24;
            pub const ENOTTY : c_int = 25;
            pub const ETXTBSY : c_int = 26;
            pub const EFBIG : c_int = 27;
            pub const ENOSPC : c_int = 28;
            pub const ESPIPE : c_int = 29;
            pub const EROFS : c_int = 30;
            pub const EMLINK : c_int = 31;
            pub const EPIPE : c_int = 32;
            pub const EDOM : c_int = 33;
            pub const ERANGE : c_int = 34;
            pub const EAGAIN : c_int = 35;
            pub const EWOULDBLOCK : c_int = 35;
            pub const EINPROGRESS : c_int = 36;
            pub const EALREADY : c_int = 37;
            pub const ENOTSOCK : c_int = 38;
            pub const EDESTADDRREQ : c_int = 39;
            pub const EMSGSIZE : c_int = 40;
            pub const EPROTOTYPE : c_int = 41;
            pub const ENOPROTOOPT : c_int = 42;
            pub const EPROTONOSUPPORT : c_int = 43;
            pub const ESOCKTNOSUPPORT : c_int = 44;
            pub const EOPNOTSUPP : c_int = 45;
            pub const EPFNOSUPPORT : c_int = 46;
            pub const EAFNOSUPPORT : c_int = 47;
            pub const EADDRINUSE : c_int = 48;
            pub const EADDRNOTAVAIL : c_int = 49;
            pub const ENETDOWN : c_int = 50;
            pub const ENETUNREACH : c_int = 51;
            pub const ENETRESET : c_int = 52;
            pub const ECONNABORTED : c_int = 53;
            pub const ECONNRESET : c_int = 54;
            pub const ENOBUFS : c_int = 55;
            pub const EISCONN : c_int = 56;
            pub const ENOTCONN : c_int = 57;
            pub const ESHUTDOWN : c_int = 58;
            pub const ETOOMANYREFS : c_int = 59;
            pub const ETIMEDOUT : c_int = 60;
            pub const ECONNREFUSED : c_int = 61;
            pub const ELOOP : c_int = 62;
            pub const ENAMETOOLONG : c_int = 63;
            pub const EHOSTDOWN : c_int = 64;
            pub const EHOSTUNREACH : c_int = 65;
            pub const ENOTEMPTY : c_int = 66;
            pub const EPROCLIM : c_int = 67;
            pub const EUSERS : c_int = 68;
            pub const EDQUOT : c_int = 69;
            pub const ESTALE : c_int = 70;
            pub const EREMOTE : c_int = 71;
            pub const EBADRPC : c_int = 72;
            pub const ERPCMISMATCH : c_int = 73;
            pub const EPROGUNAVAIL : c_int = 74;
            pub const EPROGMISMATCH : c_int = 75;
            pub const EPROCUNAVAIL : c_int = 76;
            pub const ENOLCK : c_int = 77;
            pub const ENOSYS : c_int = 78;
            pub const EFTYPE : c_int = 79;
            pub const EAUTH : c_int = 80;
            pub const ENEEDAUTH : c_int = 81;
            pub const EIDRM : c_int = 82;
            pub const ENOMSG : c_int = 83;
            pub const EOVERFLOW : c_int = 84;
            pub const ECANCELED : c_int = 85;
            pub const EILSEQ : c_int = 86;
            pub const ENOATTR : c_int = 87;
            pub const EDOOFUS : c_int = 88;
            pub const EBADMSG : c_int = 89;
            pub const EMULTIHOP : c_int = 90;
            pub const ENOLINK : c_int = 91;
            pub const EPROTO : c_int = 92;
            pub const ENOMEDIUM : c_int = 93;
            pub const EUNUSED94 : c_int = 94;
            pub const EUNUSED95 : c_int = 95;
            pub const EUNUSED96 : c_int = 96;
            pub const EUNUSED97 : c_int = 97;
            pub const EUNUSED98 : c_int = 98;
            pub const EASYNC : c_int = 99;
            pub const ELAST : c_int = 99;
        }
        pub mod posix01 {
            use types::os::arch::c95::{c_int, size_t};

            pub const F_DUPFD : c_int = 0;
            pub const F_GETFD : c_int = 1;
            pub const F_SETFD : c_int = 2;
            pub const F_GETFL : c_int = 3;
            pub const F_SETFL : c_int = 4;

            pub const SIGTRAP : c_int = 5;
            pub const SIGPIPE: c_int = 13;
            pub const SIG_IGN: size_t = 1;

            pub const GLOB_APPEND   : c_int = 0x0001;
            pub const GLOB_DOOFFS   : c_int = 0x0002;
            pub const GLOB_ERR      : c_int = 0x0004;
            pub const GLOB_MARK     : c_int = 0x0008;
            pub const GLOB_NOCHECK  : c_int = 0x0010;
            pub const GLOB_NOSORT   : c_int = 0x0020;
            pub const GLOB_NOESCAPE : c_int = 0x1000; // changed

            pub const GLOB_NOSPACE  : c_int = -1;
            pub const GLOB_ABORTED  : c_int = -2;
            pub const GLOB_NOMATCH  : c_int = -3;

            pub const POSIX_MADV_NORMAL : c_int = 0;
            pub const POSIX_MADV_RANDOM : c_int = 1;
            pub const POSIX_MADV_SEQUENTIAL : c_int = 2;
            pub const POSIX_MADV_WILLNEED : c_int = 3;
            pub const POSIX_MADV_DONTNEED : c_int = 4;

            pub const _SC_IOV_MAX : c_int = 51; // all changed...
            pub const _SC_GETGR_R_SIZE_MAX : c_int = 100;
            pub const _SC_GETPW_R_SIZE_MAX : c_int = 101;
            pub const _SC_LOGIN_NAME_MAX : c_int = 102;
            pub const _SC_MQ_PRIO_MAX : c_int = 59;
            pub const _SC_THREAD_ATTR_STACKADDR : c_int = 77;
            pub const _SC_THREAD_ATTR_STACKSIZE : c_int = 78;
            pub const _SC_THREAD_DESTRUCTOR_ITERATIONS : c_int = 80;
            pub const _SC_THREAD_KEYS_MAX : c_int = 81;
            pub const _SC_THREAD_PRIO_INHERIT : c_int = 82;
            pub const _SC_THREAD_PRIO_PROTECT : c_int = 83;
            pub const _SC_THREAD_PRIORITY_SCHEDULING : c_int = 84;
            pub const _SC_THREAD_PROCESS_SHARED : c_int = 85;
            pub const _SC_THREAD_SAFE_FUNCTIONS : c_int = 103;
            pub const _SC_THREAD_STACK_MIN : c_int = 89;
            pub const _SC_THREAD_THREADS_MAX : c_int = 90;
            pub const _SC_THREADS : c_int = 91;
            pub const _SC_TTY_NAME_MAX : c_int = 107;
            pub const _SC_ATEXIT_MAX : c_int = 46;
            pub const _SC_XOPEN_CRYPT : c_int = 117;
            pub const _SC_XOPEN_ENH_I18N : c_int = 118;
            pub const _SC_XOPEN_LEGACY : c_int = 119;
            pub const _SC_XOPEN_REALTIME : c_int = 120;
            pub const _SC_XOPEN_REALTIME_THREADS : c_int = 121;
            pub const _SC_XOPEN_SHM : c_int = 30;
            pub const _SC_XOPEN_UNIX : c_int = 123;
            pub const _SC_XOPEN_VERSION : c_int = 125;
            //pub const _SC_XOPEN_XCU_VERSION : c_int = ;

            pub const PTHREAD_CREATE_JOINABLE: c_int = 0;
            pub const PTHREAD_CREATE_DETACHED: c_int = 1;
            pub const PTHREAD_STACK_MIN: size_t = 2048;

            pub const CLOCK_REALTIME: c_int = 0;
            pub const CLOCK_MONOTONIC: c_int = 3;
        }
        pub mod posix08 {
        }
        pub mod bsd44 {
            use types::os::arch::c95::c_int;

            pub const MADV_NORMAL : c_int = 0;
            pub const MADV_RANDOM : c_int = 1;
            pub const MADV_SEQUENTIAL : c_int = 2;
            pub const MADV_WILLNEED : c_int = 3;
            pub const MADV_DONTNEED : c_int = 4;
            pub const MADV_FREE : c_int = 6; // changed
            //pub const MADV_NOSYNC : c_int = ;
            //pub const MADV_AUTOSYNC : c_int = ;
            //pub const MADV_NOCORE : c_int = ;
            //pub const MADV_CORE : c_int = ;
            //pub const MADV_PROTECT : c_int = ;

            //pub const MINCORE_INCORE : c_int =  ;
            //pub const MINCORE_REFERENCED : c_int = ;
            //pub const MINCORE_MODIFIED : c_int = ;
            //pub const MINCORE_REFERENCED_OTHER : c_int = ;
            //pub const MINCORE_MODIFIED_OTHER : c_int = ;
            //pub const MINCORE_SUPER : c_int = ;

            pub const AF_INET: c_int = 2;
            pub const AF_INET6: c_int = 24; // changed
            pub const AF_UNIX: c_int = 1;
            pub const SOCK_STREAM: c_int = 1;
            pub const SOCK_DGRAM: c_int = 2;
            pub const SOCK_RAW: c_int = 3;
            pub const IPPROTO_TCP: c_int = 6;
            pub const IPPROTO_IP: c_int = 0;
            pub const IPPROTO_IPV6: c_int = 41;
            pub const IP_MULTICAST_TTL: c_int = 10;
            pub const IP_MULTICAST_LOOP: c_int = 11;
            pub const IP_TTL: c_int = 4;
            pub const IP_HDRINCL: c_int = 2;
            pub const IP_ADD_MEMBERSHIP: c_int = 12;
            pub const IP_DROP_MEMBERSHIP: c_int = 13;
            // don't exist, keep same as IP_ADD_MEMBERSHIP
            pub const IPV6_ADD_MEMBERSHIP: c_int = 12;
            // don't exist, keep same as IP_DROP_MEMBERSHIP
            pub const IPV6_DROP_MEMBERSHIP: c_int = 13;

            pub const TCP_NODELAY: c_int = 1;
            //pub const TCP_KEEPIDLE: c_int = ;
            pub const SOL_SOCKET: c_int = 0xffff;
            pub const SO_KEEPALIVE: c_int = 0x0008;
            pub const SO_BROADCAST: c_int = 0x0020;
            pub const SO_REUSEADDR: c_int = 0x0004;
            pub const SO_ERROR: c_int = 0x1007;

            pub const IFF_LOOPBACK: c_int = 0x8;

            pub const SHUT_RD: c_int = 0;
            pub const SHUT_WR: c_int = 1;
            pub const SHUT_RDWR: c_int = 2;
        }
        pub mod extra {
            use types::os::arch::c95::c_int;

            pub const O_SYNC : c_int = 128;
            pub const O_NONBLOCK : c_int = 4;
            pub const CTL_KERN: c_int = 1;
            pub const KERN_PROC: c_int = 66;

            pub const MAP_COPY : c_int = 0x0002;
            pub const MAP_RENAME : c_int = 0x0000; // changed
            pub const MAP_NORESERVE : c_int = 0x0000; // changed
            pub const MAP_HASSEMAPHORE : c_int = 0x0000; // changed
            //pub const MAP_STACK : c_int = ;
            //pub const MAP_NOSYNC : c_int = ;
            //pub const MAP_NOCORE : c_int = ;

            pub const IPPROTO_RAW : c_int = 255;
        }
        pub mod sysconf {
            use types::os::arch::c95::c_int;

            pub const _SC_ARG_MAX : c_int = 1;
            pub const _SC_CHILD_MAX : c_int = 2;
            pub const _SC_CLK_TCK : c_int = 3;
            pub const _SC_NGROUPS_MAX : c_int = 4;
            pub const _SC_OPEN_MAX : c_int = 5;
            pub const _SC_JOB_CONTROL : c_int = 6;
            pub const _SC_SAVED_IDS : c_int = 7;
            pub const _SC_VERSION : c_int = 8;
            pub const _SC_BC_BASE_MAX : c_int = 9;
            pub const _SC_BC_DIM_MAX : c_int = 10;
            pub const _SC_BC_SCALE_MAX : c_int = 11;
            pub const _SC_BC_STRING_MAX : c_int = 12;
            pub const _SC_COLL_WEIGHTS_MAX : c_int = 13;
            pub const _SC_EXPR_NEST_MAX : c_int = 14;
            pub const _SC_LINE_MAX : c_int = 15;
            pub const _SC_RE_DUP_MAX : c_int = 16;
            pub const _SC_2_VERSION : c_int = 17;
            pub const _SC_2_C_BIND : c_int = 18;
            pub const _SC_2_C_DEV : c_int = 19;
            pub const _SC_2_CHAR_TERM : c_int = 20;
            pub const _SC_2_FORT_DEV : c_int = 21;
            pub const _SC_2_FORT_RUN : c_int = 22;
            pub const _SC_2_LOCALEDEF : c_int = 23;
            pub const _SC_2_SW_DEV : c_int = 24;
            pub const _SC_2_UPE : c_int = 25;
            pub const _SC_STREAM_MAX : c_int = 26;
            pub const _SC_TZNAME_MAX : c_int = 27;
            pub const _SC_ASYNCHRONOUS_IO : c_int = 45; // changed...
            pub const _SC_MAPPED_FILES : c_int = 53;
            pub const _SC_MEMLOCK : c_int = 54;
            pub const _SC_MEMLOCK_RANGE : c_int = 55;
            pub const _SC_MEMORY_PROTECTION : c_int = 56;
            pub const _SC_MESSAGE_PASSING : c_int = 57;
            pub const _SC_PRIORITIZED_IO : c_int = 60;
            pub const _SC_PRIORITY_SCHEDULING : c_int = 61;
            pub const _SC_REALTIME_SIGNALS : c_int = 64;
            pub const _SC_SEMAPHORES : c_int = 67;
            pub const _SC_FSYNC : c_int = 29;
            pub const _SC_SHARED_MEMORY_OBJECTS : c_int = 68;
            pub const _SC_SYNCHRONIZED_IO : c_int = 75;
            pub const _SC_TIMERS : c_int = 94; // ...changed
            pub const _SC_AIO_LISTIO_MAX : c_int = 42;
            pub const _SC_AIO_MAX : c_int = 43;
            pub const _SC_AIO_PRIO_DELTA_MAX : c_int = 44;
            pub const _SC_DELAYTIMER_MAX : c_int = 50; // changed...
            pub const _SC_MQ_OPEN_MAX : c_int = 58;
            pub const _SC_PAGESIZE : c_int = 28;
            pub const _SC_RTSIG_MAX : c_int = 66;
            pub const _SC_SEM_NSEMS_MAX : c_int = 31;
            pub const _SC_SEM_VALUE_MAX : c_int = 32;
            pub const _SC_SIGQUEUE_MAX : c_int = 70;
            pub const _SC_TIMER_MAX : c_int = 93;
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub mod os {
        pub mod c95 {
            use types::os::arch::c95::{c_int, c_uint};

            pub const EXIT_FAILURE : c_int = 1;
            pub const EXIT_SUCCESS : c_int = 0;
            pub const RAND_MAX : c_int = 2147483647;
            pub const EOF : c_int = -1;
            pub const SEEK_SET : c_int = 0;
            pub const SEEK_CUR : c_int = 1;
            pub const SEEK_END : c_int = 2;
            pub const _IOFBF : c_int = 0;
            pub const _IONBF : c_int = 2;
            pub const _IOLBF : c_int = 1;
            pub const BUFSIZ : c_uint = 1024_u32;
            pub const FOPEN_MAX : c_uint = 20_u32;
            pub const FILENAME_MAX : c_uint = 1024_u32;
            pub const L_tmpnam : c_uint = 1024_u32;
            pub const TMP_MAX : c_uint = 308915776_u32;
        }
        pub mod c99 {
        }
        pub mod posix88 {
            use types::common::c95::c_void;
            use types::os::arch::c95::c_int;
            use types::os::arch::posix88::mode_t;

            pub const O_RDONLY : c_int = 0;
            pub const O_WRONLY : c_int = 1;
            pub const O_RDWR : c_int = 2;
            pub const O_APPEND : c_int = 8;
            pub const O_CREAT : c_int = 512;
            pub const O_EXCL : c_int = 2048;
            pub const O_TRUNC : c_int = 1024;
            pub const S_IFIFO : mode_t = 4096;
            pub const S_IFCHR : mode_t = 8192;
            pub const S_IFBLK : mode_t = 24576;
            pub const S_IFDIR : mode_t = 16384;
            pub const S_IFREG : mode_t = 32768;
            pub const S_IFLNK : mode_t = 40960;
            pub const S_IFMT : mode_t = 61440;
            pub const S_IEXEC : mode_t = 64;
            pub const S_IWRITE : mode_t = 128;
            pub const S_IREAD : mode_t = 256;
            pub const S_IRWXU : mode_t = 448;
            pub const S_IXUSR : mode_t = 64;
            pub const S_IWUSR : mode_t = 128;
            pub const S_IRUSR : mode_t = 256;
            pub const F_OK : c_int = 0;
            pub const R_OK : c_int = 4;
            pub const W_OK : c_int = 2;
            pub const X_OK : c_int = 1;
            pub const STDIN_FILENO : c_int = 0;
            pub const STDOUT_FILENO : c_int = 1;
            pub const STDERR_FILENO : c_int = 2;
            pub const F_LOCK : c_int = 1;
            pub const F_TEST : c_int = 3;
            pub const F_TLOCK : c_int = 2;
            pub const F_ULOCK : c_int = 0;
            pub const SIGHUP : c_int = 1;
            pub const SIGINT : c_int = 2;
            pub const SIGQUIT : c_int = 3;
            pub const SIGILL : c_int = 4;
            pub const SIGABRT : c_int = 6;
            pub const SIGFPE : c_int = 8;
            pub const SIGKILL : c_int = 9;
            pub const SIGSEGV : c_int = 11;
            pub const SIGPIPE : c_int = 13;
            pub const SIGALRM : c_int = 14;
            pub const SIGTERM : c_int = 15;

            pub const PROT_NONE : c_int = 0;
            pub const PROT_READ : c_int = 1;
            pub const PROT_WRITE : c_int = 2;
            pub const PROT_EXEC : c_int = 4;

            pub const MAP_FILE : c_int = 0x0000;
            pub const MAP_SHARED : c_int = 0x0001;
            pub const MAP_PRIVATE : c_int = 0x0002;
            pub const MAP_FIXED : c_int = 0x0010;
            pub const MAP_ANON : c_int = 0x1000;

            pub const MAP_FAILED : *mut c_void = -1 as *mut c_void;

            pub const MCL_CURRENT : c_int = 0x0001;
            pub const MCL_FUTURE : c_int = 0x0002;

            pub const MS_ASYNC : c_int = 0x0001;
            pub const MS_INVALIDATE : c_int = 0x0002;
            pub const MS_SYNC : c_int = 0x0010;

            pub const MS_KILLPAGES : c_int = 0x0004;
            pub const MS_DEACTIVATE : c_int = 0x0008;

            pub const EPERM : c_int = 1;
            pub const ENOENT : c_int = 2;
            pub const ESRCH : c_int = 3;
            pub const EINTR : c_int = 4;
            pub const EIO : c_int = 5;
            pub const ENXIO : c_int = 6;
            pub const E2BIG : c_int = 7;
            pub const ENOEXEC : c_int = 8;
            pub const EBADF : c_int = 9;
            pub const ECHILD : c_int = 10;
            pub const EDEADLK : c_int = 11;
            pub const ENOMEM : c_int = 12;
            pub const EACCES : c_int = 13;
            pub const EFAULT : c_int = 14;
            pub const ENOTBLK : c_int = 15;
            pub const EBUSY : c_int = 16;
            pub const EEXIST : c_int = 17;
            pub const EXDEV : c_int = 18;
            pub const ENODEV : c_int = 19;
            pub const ENOTDIR : c_int = 20;
            pub const EISDIR : c_int = 21;
            pub const EINVAL : c_int = 22;
            pub const ENFILE : c_int = 23;
            pub const EMFILE : c_int = 24;
            pub const ENOTTY : c_int = 25;
            pub const ETXTBSY : c_int = 26;
            pub const EFBIG : c_int = 27;
            pub const ENOSPC : c_int = 28;
            pub const ESPIPE : c_int = 29;
            pub const EROFS : c_int = 30;
            pub const EMLINK : c_int = 31;
            pub const EPIPE : c_int = 32;
            pub const EDOM : c_int = 33;
            pub const ERANGE : c_int = 34;
            pub const EAGAIN : c_int = 35;
            pub const EWOULDBLOCK : c_int = EAGAIN;
            pub const EINPROGRESS : c_int = 36;
            pub const EALREADY : c_int = 37;
            pub const ENOTSOCK : c_int = 38;
            pub const EDESTADDRREQ : c_int = 39;
            pub const EMSGSIZE : c_int = 40;
            pub const EPROTOTYPE : c_int = 41;
            pub const ENOPROTOOPT : c_int = 42;
            pub const EPROTONOSUPPORT : c_int = 43;
            pub const ESOCKTNOSUPPORT : c_int = 44;
            pub const ENOTSUP : c_int = 45;
            pub const EPFNOSUPPORT : c_int = 46;
            pub const EAFNOSUPPORT : c_int = 47;
            pub const EADDRINUSE : c_int = 48;
            pub const EADDRNOTAVAIL : c_int = 49;
            pub const ENETDOWN : c_int = 50;
            pub const ENETUNREACH : c_int = 51;
            pub const ENETRESET : c_int = 52;
            pub const ECONNABORTED : c_int = 53;
            pub const ECONNRESET : c_int = 54;
            pub const ENOBUFS : c_int = 55;
            pub const EISCONN : c_int = 56;
            pub const ENOTCONN : c_int = 57;
            pub const ESHUTDOWN : c_int = 58;
            pub const ETOOMANYREFS : c_int = 59;
            pub const ETIMEDOUT : c_int = 60;
            pub const ECONNREFUSED : c_int = 61;
            pub const ELOOP : c_int = 62;
            pub const ENAMETOOLONG : c_int = 63;
            pub const EHOSTDOWN : c_int = 64;
            pub const EHOSTUNREACH : c_int = 65;
            pub const ENOTEMPTY : c_int = 66;
            pub const EPROCLIM : c_int = 67;
            pub const EUSERS : c_int = 68;
            pub const EDQUOT : c_int = 69;
            pub const ESTALE : c_int = 70;
            pub const EREMOTE : c_int = 71;
            pub const EBADRPC : c_int = 72;
            pub const ERPCMISMATCH : c_int = 73;
            pub const EPROGUNAVAIL : c_int = 74;
            pub const EPROGMISMATCH : c_int = 75;
            pub const EPROCUNAVAIL : c_int = 76;
            pub const ENOLCK : c_int = 77;
            pub const ENOSYS : c_int = 78;
            pub const EFTYPE : c_int = 79;
            pub const EAUTH : c_int = 80;
            pub const ENEEDAUTH : c_int = 81;
            pub const EPWROFF : c_int = 82;
            pub const EDEVERR : c_int = 83;
            pub const EOVERFLOW : c_int = 84;
            pub const EBADEXEC : c_int = 85;
            pub const EBADARCH : c_int = 86;
            pub const ESHLIBVERS : c_int = 87;
            pub const EBADMACHO : c_int = 88;
            pub const ECANCELED : c_int = 89;
            pub const EIDRM : c_int = 90;
            pub const ENOMSG : c_int = 91;
            pub const EILSEQ : c_int = 92;
            pub const ENOATTR : c_int = 93;
            pub const EBADMSG : c_int = 94;
            pub const EMULTIHOP : c_int = 95;
            pub const ENODATA : c_int = 96;
            pub const ENOLINK : c_int = 97;
            pub const ENOSR : c_int = 98;
            pub const ENOSTR : c_int = 99;
            pub const EPROTO : c_int = 100;
            pub const ETIME : c_int = 101;
            pub const EOPNOTSUPP : c_int = 102;
            pub const ENOPOLICY : c_int = 103;
            pub const ENOTRECOVERABLE : c_int = 104;
            pub const EOWNERDEAD : c_int = 105;
            pub const EQFULL : c_int = 106;
            pub const ELAST : c_int = 106;
        }
        pub mod posix01 {
            use types::os::arch::c95::{c_int, size_t};

            pub const F_DUPFD : c_int = 0;
            pub const F_GETFD : c_int = 1;
            pub const F_SETFD : c_int = 2;
            pub const F_GETFL : c_int = 3;
            pub const F_SETFL : c_int = 4;

            pub const SIGTRAP : c_int = 5;
            pub const SIGPIPE: c_int = 13;
            pub const SIG_IGN: size_t = 1;

            pub const GLOB_APPEND   : c_int = 0x0001;
            pub const GLOB_DOOFFS   : c_int = 0x0002;
            pub const GLOB_ERR      : c_int = 0x0004;
            pub const GLOB_MARK     : c_int = 0x0008;
            pub const GLOB_NOCHECK  : c_int = 0x0010;
            pub const GLOB_NOSORT   : c_int = 0x0020;
            pub const GLOB_NOESCAPE : c_int = 0x2000;

            pub const GLOB_NOSPACE  : c_int = -1;
            pub const GLOB_ABORTED  : c_int = -2;
            pub const GLOB_NOMATCH  : c_int = -3;

            pub const POSIX_MADV_NORMAL : c_int = 0;
            pub const POSIX_MADV_RANDOM : c_int = 1;
            pub const POSIX_MADV_SEQUENTIAL : c_int = 2;
            pub const POSIX_MADV_WILLNEED : c_int = 3;
            pub const POSIX_MADV_DONTNEED : c_int = 4;

            pub const _SC_IOV_MAX : c_int = 56;
            pub const _SC_GETGR_R_SIZE_MAX : c_int = 70;
            pub const _SC_GETPW_R_SIZE_MAX : c_int = 71;
            pub const _SC_LOGIN_NAME_MAX : c_int = 73;
            pub const _SC_MQ_PRIO_MAX : c_int = 75;
            pub const _SC_THREAD_ATTR_STACKADDR : c_int = 82;
            pub const _SC_THREAD_ATTR_STACKSIZE : c_int = 83;
            pub const _SC_THREAD_DESTRUCTOR_ITERATIONS : c_int = 85;
            pub const _SC_THREAD_KEYS_MAX : c_int = 86;
            pub const _SC_THREAD_PRIO_INHERIT : c_int = 87;
            pub const _SC_THREAD_PRIO_PROTECT : c_int = 88;
            pub const _SC_THREAD_PRIORITY_SCHEDULING : c_int = 89;
            pub const _SC_THREAD_PROCESS_SHARED : c_int = 90;
            pub const _SC_THREAD_SAFE_FUNCTIONS : c_int = 91;
            pub const _SC_THREAD_STACK_MIN : c_int = 93;
            pub const _SC_THREAD_THREADS_MAX : c_int = 94;
            pub const _SC_THREADS : c_int = 96;
            pub const _SC_TTY_NAME_MAX : c_int = 101;
            pub const _SC_ATEXIT_MAX : c_int = 107;
            pub const _SC_XOPEN_CRYPT : c_int = 108;
            pub const _SC_XOPEN_ENH_I18N : c_int = 109;
            pub const _SC_XOPEN_LEGACY : c_int = 110;
            pub const _SC_XOPEN_REALTIME : c_int = 111;
            pub const _SC_XOPEN_REALTIME_THREADS : c_int = 112;
            pub const _SC_XOPEN_SHM : c_int = 113;
            pub const _SC_XOPEN_UNIX : c_int = 115;
            pub const _SC_XOPEN_VERSION : c_int = 116;
            pub const _SC_XOPEN_XCU_VERSION : c_int = 121;

            pub const PTHREAD_CREATE_JOINABLE: c_int = 1;
            pub const PTHREAD_CREATE_DETACHED: c_int = 2;
            pub const PTHREAD_STACK_MIN: size_t = 8192;
        }
        pub mod posix08 {
        }
        pub mod bsd44 {
            use types::os::arch::c95::c_int;

            pub const MADV_NORMAL : c_int = 0;
            pub const MADV_RANDOM : c_int = 1;
            pub const MADV_SEQUENTIAL : c_int = 2;
            pub const MADV_WILLNEED : c_int = 3;
            pub const MADV_DONTNEED : c_int = 4;
            pub const MADV_FREE : c_int = 5;
            pub const MADV_ZERO_WIRED_PAGES : c_int = 6;
            pub const MADV_FREE_REUSABLE : c_int = 7;
            pub const MADV_FREE_REUSE : c_int = 8;
            pub const MADV_CAN_REUSE : c_int = 9;

            pub const MINCORE_INCORE : c_int =  0x1;
            pub const MINCORE_REFERENCED : c_int = 0x2;
            pub const MINCORE_MODIFIED : c_int = 0x4;
            pub const MINCORE_REFERENCED_OTHER : c_int = 0x8;
            pub const MINCORE_MODIFIED_OTHER : c_int = 0x10;

            pub const AF_UNIX: c_int = 1;
            pub const AF_INET: c_int = 2;
            pub const AF_INET6: c_int = 30;
            pub const SOCK_STREAM: c_int = 1;
            pub const SOCK_DGRAM: c_int = 2;
            pub const SOCK_RAW: c_int = 3;
            pub const IPPROTO_TCP: c_int = 6;
            pub const IPPROTO_IP: c_int = 0;
            pub const IPPROTO_IPV6: c_int = 41;
            pub const IP_MULTICAST_TTL: c_int = 10;
            pub const IP_MULTICAST_LOOP: c_int = 11;
            pub const IP_TTL: c_int = 4;
            pub const IP_HDRINCL: c_int = 2;
            pub const IP_ADD_MEMBERSHIP: c_int = 12;
            pub const IP_DROP_MEMBERSHIP: c_int = 13;
            pub const IPV6_ADD_MEMBERSHIP: c_int = 12;
            pub const IPV6_DROP_MEMBERSHIP: c_int = 13;

            pub const TCP_NODELAY: c_int = 0x01;
            pub const TCP_KEEPALIVE: c_int = 0x10;
            pub const SOL_SOCKET: c_int = 0xffff;
            pub const SO_KEEPALIVE: c_int = 0x0008;
            pub const SO_BROADCAST: c_int = 0x0020;
            pub const SO_REUSEADDR: c_int = 0x0004;
            pub const SO_ERROR: c_int = 0x1007;

            pub const IFF_LOOPBACK: c_int = 0x8;

            pub const SHUT_RD: c_int = 0;
            pub const SHUT_WR: c_int = 1;
            pub const SHUT_RDWR: c_int = 2;
        }
        pub mod extra {
            use types::os::arch::c95::c_int;

            pub const O_DSYNC : c_int = 4194304;
            pub const O_SYNC : c_int = 128;
            pub const O_NONBLOCK : c_int = 4;
            pub const F_FULLFSYNC : c_int = 51;

            pub const MAP_COPY : c_int = 0x0002;
            pub const MAP_RENAME : c_int = 0x0020;
            pub const MAP_NORESERVE : c_int = 0x0040;
            pub const MAP_NOEXTEND : c_int = 0x0100;
            pub const MAP_HASSEMAPHORE : c_int = 0x0200;
            pub const MAP_NOCACHE : c_int = 0x0400;
            pub const MAP_JIT : c_int = 0x0800;
            pub const MAP_STACK : c_int = 0;

            pub const IPPROTO_RAW : c_int = 255;
        }
        pub mod sysconf {
            use types::os::arch::c95::c_int;

            pub const _SC_ARG_MAX : c_int = 1;
            pub const _SC_CHILD_MAX : c_int = 2;
            pub const _SC_CLK_TCK : c_int = 3;
            pub const _SC_NGROUPS_MAX : c_int = 4;
            pub const _SC_OPEN_MAX : c_int = 5;
            pub const _SC_JOB_CONTROL : c_int = 6;
            pub const _SC_SAVED_IDS : c_int = 7;
            pub const _SC_VERSION : c_int = 8;
            pub const _SC_BC_BASE_MAX : c_int = 9;
            pub const _SC_BC_DIM_MAX : c_int = 10;
            pub const _SC_BC_SCALE_MAX : c_int = 11;
            pub const _SC_BC_STRING_MAX : c_int = 12;
            pub const _SC_COLL_WEIGHTS_MAX : c_int = 13;
            pub const _SC_EXPR_NEST_MAX : c_int = 14;
            pub const _SC_LINE_MAX : c_int = 15;
            pub const _SC_RE_DUP_MAX : c_int = 16;
            pub const _SC_2_VERSION : c_int = 17;
            pub const _SC_2_C_BIND : c_int = 18;
            pub const _SC_2_C_DEV : c_int = 19;
            pub const _SC_2_CHAR_TERM : c_int = 20;
            pub const _SC_2_FORT_DEV : c_int = 21;
            pub const _SC_2_FORT_RUN : c_int = 22;
            pub const _SC_2_LOCALEDEF : c_int = 23;
            pub const _SC_2_SW_DEV : c_int = 24;
            pub const _SC_2_UPE : c_int = 25;
            pub const _SC_STREAM_MAX : c_int = 26;
            pub const _SC_TZNAME_MAX : c_int = 27;
            pub const _SC_ASYNCHRONOUS_IO : c_int = 28;
            pub const _SC_PAGESIZE : c_int = 29;
            pub const _SC_MEMLOCK : c_int = 30;
            pub const _SC_MEMLOCK_RANGE : c_int = 31;
            pub const _SC_MEMORY_PROTECTION : c_int = 32;
            pub const _SC_MESSAGE_PASSING : c_int = 33;
            pub const _SC_PRIORITIZED_IO : c_int = 34;
            pub const _SC_PRIORITY_SCHEDULING : c_int = 35;
            pub const _SC_REALTIME_SIGNALS : c_int = 36;
            pub const _SC_SEMAPHORES : c_int = 37;
            pub const _SC_FSYNC : c_int = 38;
            pub const _SC_SHARED_MEMORY_OBJECTS : c_int = 39;
            pub const _SC_SYNCHRONIZED_IO : c_int = 40;
            pub const _SC_TIMERS : c_int = 41;
            pub const _SC_AIO_LISTIO_MAX : c_int = 42;
            pub const _SC_AIO_MAX : c_int = 43;
            pub const _SC_AIO_PRIO_DELTA_MAX : c_int = 44;
            pub const _SC_DELAYTIMER_MAX : c_int = 45;
            pub const _SC_MQ_OPEN_MAX : c_int = 46;
            pub const _SC_MAPPED_FILES : c_int = 47;
            pub const _SC_RTSIG_MAX : c_int = 48;
            pub const _SC_SEM_NSEMS_MAX : c_int = 49;
            pub const _SC_SEM_VALUE_MAX : c_int = 50;
            pub const _SC_SIGQUEUE_MAX : c_int = 51;
            pub const _SC_TIMER_MAX : c_int = 52;
            pub const _SC_XBS5_ILP32_OFF32 : c_int = 122;
            pub const _SC_XBS5_ILP32_OFFBIG : c_int = 123;
            pub const _SC_XBS5_LP64_OFF64 : c_int = 124;
            pub const _SC_XBS5_LPBIG_OFFBIG : c_int = 125;
        }
    }
}


pub mod funcs {
    // Thankfully most of c95 is universally available and does not vary by OS
    // or anything. The same is not true of POSIX.

    pub mod c95 {
        pub mod ctype {
            use types::os::arch::c95::{c_char, c_int};

            extern {
                pub fn isalnum(c: c_int) -> c_int;
                pub fn isalpha(c: c_int) -> c_int;
                pub fn iscntrl(c: c_int) -> c_int;
                pub fn isdigit(c: c_int) -> c_int;
                pub fn isgraph(c: c_int) -> c_int;
                pub fn islower(c: c_int) -> c_int;
                pub fn isprint(c: c_int) -> c_int;
                pub fn ispunct(c: c_int) -> c_int;
                pub fn isspace(c: c_int) -> c_int;
                pub fn isupper(c: c_int) -> c_int;
                pub fn isxdigit(c: c_int) -> c_int;
                pub fn tolower(c: c_char) -> c_char;
                pub fn toupper(c: c_char) -> c_char;
            }
        }

        pub mod stdio {
            use types::common::c95::{FILE, c_void, fpos_t};
            use types::os::arch::c95::{c_char, c_int, c_long, size_t};

            extern {
                pub fn fopen(filename: *const c_char,
                             mode: *const c_char) -> *mut FILE;
                pub fn freopen(filename: *const c_char, mode: *const c_char,
                               file: *mut FILE)
                               -> *mut FILE;
                pub fn fflush(file: *mut FILE) -> c_int;
                pub fn fclose(file: *mut FILE) -> c_int;
                pub fn remove(filename: *const c_char) -> c_int;
                pub fn rename(oldname: *const c_char,
                              newname: *const c_char) -> c_int;
                pub fn tmpfile() -> *mut FILE;
                pub fn setvbuf(stream: *mut FILE,
                               buffer: *mut c_char,
                               mode: c_int,
                               size: size_t)
                               -> c_int;
                pub fn setbuf(stream: *mut FILE, buf: *mut c_char);
                // Omitted: printf and scanf variants.
                pub fn fgetc(stream: *mut FILE) -> c_int;
                pub fn fgets(buf: *mut c_char, n: c_int, stream: *mut FILE)
                             -> *mut c_char;
                pub fn fputc(c: c_int, stream: *mut FILE) -> c_int;
                pub fn fputs(s: *const c_char, stream: *mut FILE)-> c_int;
                // Omitted: getc, getchar (might be macros).

                // Omitted: gets, so ridiculously unsafe that it should not
                // survive.

                // Omitted: putc, putchar (might be macros).
                pub fn puts(s: *const c_char) -> c_int;
                pub fn ungetc(c: c_int, stream: *mut FILE) -> c_int;
                pub fn fread(ptr: *mut c_void,
                             size: size_t,
                             nobj: size_t,
                             stream: *mut FILE)
                             -> size_t;
                pub fn fwrite(ptr: *const c_void,
                              size: size_t,
                              nobj: size_t,
                              stream: *mut FILE)
                              -> size_t;
                pub fn fseek(stream: *mut FILE, offset: c_long, whence: c_int)
                             -> c_int;
                pub fn ftell(stream: *mut FILE) -> c_long;
                pub fn rewind(stream: *mut FILE);
                pub fn fgetpos(stream: *mut FILE, ptr: *mut fpos_t) -> c_int;
                pub fn fsetpos(stream: *mut FILE, ptr: *mut fpos_t) -> c_int;
                pub fn feof(stream: *mut FILE) -> c_int;
                pub fn ferror(stream: *mut FILE) -> c_int;
                pub fn perror(s: *const c_char);
            }
        }

        pub mod stdlib {
            use types::common::c95::c_void;
            use types::os::arch::c95::{c_char, c_double, c_int};
            use types::os::arch::c95::{c_long, c_uint, c_ulong};
            use types::os::arch::c95::{size_t};

            extern {
                pub fn abs(i: c_int) -> c_int;
                pub fn labs(i: c_long) -> c_long;
                // Omitted: div, ldiv (return pub type incomplete).
                pub fn atof(s: *const c_char) -> c_double;
                pub fn atoi(s: *const c_char) -> c_int;
                pub fn strtod(s: *const c_char,
                              endp: *mut *mut c_char) -> c_double;
                pub fn strtol(s: *const c_char,
                              endp: *mut *mut c_char, base: c_int) -> c_long;
                pub fn strtoul(s: *const c_char, endp: *mut *mut c_char,
                               base: c_int) -> c_ulong;
                pub fn calloc(nobj: size_t, size: size_t) -> *mut c_void;
                pub fn malloc(size: size_t) -> *mut c_void;
                pub fn realloc(p: *mut c_void, size: size_t) -> *mut c_void;
                pub fn free(p: *mut c_void);

                /// Exits the running program in a possibly dangerous manner.
                ///
                /// # Unsafety
                ///
                /// While this forces your program to exit, it does so in a way that has
                /// consequences. This will skip all unwinding code, which means that anything
                /// relying on unwinding for cleanup (such as flushing and closing a buffer to a
                /// file) may act in an unexpected way.
                ///
                /// # Examples
                ///
                /// ```no_run
                /// extern crate libc;
                ///
                /// fn main() {
                ///     unsafe {
                ///         libc::exit(1);
                ///     }
                /// }
                /// ```
                pub fn exit(status: c_int) -> !;
                pub fn _exit(status: c_int) -> !;
                pub fn atexit(cb: extern fn()) -> c_int;
                pub fn system(s: *const c_char) -> c_int;
                pub fn getenv(s: *const c_char) -> *mut c_char;
                // Omitted: bsearch, qsort
                pub fn rand() -> c_int;
                pub fn srand(seed: c_uint);
            }
        }

        pub mod string {
            use types::common::c95::c_void;
            use types::os::arch::c95::{c_char, c_int, size_t};
            use types::os::arch::c95::{wchar_t};

            extern {
                pub fn strcpy(dst: *mut c_char,
                              src: *const c_char) -> *mut c_char;
                pub fn strncpy(dst: *mut c_char, src: *const c_char, n: size_t)
                               -> *mut c_char;
                pub fn strcat(s: *mut c_char, ct: *const c_char) -> *mut c_char;
                pub fn strncat(s: *mut c_char, ct: *const c_char,
                               n: size_t) -> *mut c_char;
                pub fn strcmp(cs: *const c_char, ct: *const c_char) -> c_int;
                pub fn strncmp(cs: *const c_char, ct: *const c_char,
                               n: size_t) -> c_int;
                pub fn strcoll(cs: *const c_char, ct: *const c_char) -> c_int;
                pub fn strchr(cs: *const c_char, c: c_int) -> *mut c_char;
                pub fn strrchr(cs: *const c_char, c: c_int) -> *mut c_char;
                pub fn strspn(cs: *const c_char, ct: *const c_char) -> size_t;
                pub fn strcspn(cs: *const c_char, ct: *const c_char) -> size_t;
                pub fn strpbrk(cs: *const c_char,
                               ct: *const c_char) -> *mut c_char;
                pub fn strstr(cs: *const c_char,
                              ct: *const c_char) -> *mut c_char;
                pub fn strlen(cs: *const c_char) -> size_t;
                pub fn strerror(n: c_int) -> *mut c_char;
                pub fn strtok(s: *mut c_char, t: *const c_char) -> *mut c_char;
                pub fn strxfrm(s: *mut c_char, ct: *const c_char,
                               n: size_t) -> size_t;
                pub fn wcslen(buf: *const wchar_t) -> size_t;

                // Omitted: memcpy, memmove, memset (provided by LLVM)

                // These are fine to execute on the Rust stack. They must be,
                // in fact, because LLVM generates calls to them!
                pub fn memcmp(cx: *const c_void, ct: *const c_void,
                              n: size_t) -> c_int;
                pub fn memchr(cx: *const c_void, c: c_int,
                              n: size_t) -> *mut c_void;
            }
        }
    }

    // Microsoft helpfully underscore-qualifies all of its POSIX-like symbols
    // to make sure you don't use them accidentally. It also randomly deviates
    // from the exact signatures you might otherwise expect, and omits much,
    // so be careful when trying to write portable code; it won't always work
    // with the same POSIX functions and types as other platforms.

    #[cfg(target_os = "windows")]
    pub mod posix88 {
        pub mod stat_ {
            use types::os::common::posix01::{stat, utimbuf};
            use types::os::arch::c95::{c_int, c_char, wchar_t};

            extern {
                #[link_name = "_chmod"]
                pub fn chmod(path: *const c_char, mode: c_int) -> c_int;
                #[link_name = "_wchmod"]
                pub fn wchmod(path: *const wchar_t, mode: c_int) -> c_int;
                #[link_name = "_mkdir"]
                pub fn mkdir(path: *const c_char) -> c_int;
                #[link_name = "_wrmdir"]
                pub fn wrmdir(path: *const wchar_t) -> c_int;
                #[link_name = "_fstat64"]
                pub fn fstat(fildes: c_int, buf: *mut stat) -> c_int;
                #[link_name = "_stat64"]
                pub fn stat(path: *const c_char, buf: *mut stat) -> c_int;
                #[link_name = "_wstat64"]
                pub fn wstat(path: *const wchar_t, buf: *mut stat) -> c_int;
                #[link_name = "_wutime64"]
                pub fn wutime(file: *const wchar_t, buf: *mut utimbuf) -> c_int;
            }
        }

        pub mod stdio {
            use types::common::c95::FILE;
            use types::os::arch::c95::{c_int, c_char};

            extern {
                #[link_name = "_popen"]
                pub fn popen(command: *const c_char,
                             mode: *const c_char) -> *mut FILE;
                #[link_name = "_pclose"]
                pub fn pclose(stream: *mut FILE) -> c_int;
                #[link_name = "_fdopen"]
                pub fn fdopen(fd: c_int, mode: *const c_char) -> *mut FILE;
                #[link_name = "_fileno"]
                pub fn fileno(stream: *mut FILE) -> c_int;
            }
        }

        pub mod fcntl {
            use types::os::arch::c95::{c_int, c_char, wchar_t};
            extern {
                #[link_name = "_open"]
                pub fn open(path: *const c_char, oflag: c_int, mode: c_int)
                            -> c_int;
                #[link_name = "_wopen"]
                pub fn wopen(path: *const wchar_t, oflag: c_int, mode: c_int)
                            -> c_int;
                #[link_name = "_creat"]
                pub fn creat(path: *const c_char, mode: c_int) -> c_int;
            }
        }

        pub mod dirent {
            // Not supplied at all.
        }

        pub mod unistd {
            use types::common::c95::c_void;
            use types::os::arch::c95::{c_int, c_uint, c_char,
                                             c_long, size_t};
            use types::os::arch::c99::intptr_t;

            extern {
                #[link_name = "_access"]
                pub fn access(path: *const c_char, amode: c_int) -> c_int;
                #[link_name = "_chdir"]
                pub fn chdir(dir: *const c_char) -> c_int;
                #[link_name = "_close"]
                pub fn close(fd: c_int) -> c_int;
                #[link_name = "_dup"]
                pub fn dup(fd: c_int) -> c_int;
                #[link_name = "_dup2"]
                pub fn dup2(src: c_int, dst: c_int) -> c_int;
                #[link_name = "_execv"]
                pub fn execv(prog: *const c_char,
                             argv: *mut *const c_char) -> intptr_t;
                #[link_name = "_execve"]
                pub fn execve(prog: *const c_char, argv: *mut *const c_char,
                              envp: *mut *const c_char)
                              -> c_int;
                #[link_name = "_execvp"]
                pub fn execvp(c: *const c_char,
                              argv: *mut *const c_char) -> c_int;
                #[link_name = "_execvpe"]
                pub fn execvpe(c: *const c_char, argv: *mut *const c_char,
                               envp: *mut *const c_char) -> c_int;
                #[link_name = "_getcwd"]
                pub fn getcwd(buf: *mut c_char, size: size_t) -> *mut c_char;
                #[link_name = "_getpid"]
                pub fn getpid() -> c_int;
                #[link_name = "_isatty"]
                pub fn isatty(fd: c_int) -> c_int;
                #[link_name = "_lseek"]
                pub fn lseek(fd: c_int, offset: c_long, origin: c_int)
                             -> c_long;
                #[link_name = "_pipe"]
                pub fn pipe(fds: *mut c_int, psize: c_uint, textmode: c_int)
                            -> c_int;
                #[link_name = "_read"]
                pub fn read(fd: c_int, buf: *mut c_void, count: c_uint)
                            -> c_int;
                #[link_name = "_rmdir"]
                pub fn rmdir(path: *const c_char) -> c_int;
                #[link_name = "_unlink"]
                pub fn unlink(c: *const c_char) -> c_int;
                #[link_name = "_write"]
                pub fn write(fd: c_int, buf: *const c_void,
                             count: c_uint) -> c_int;
            }
        }

        pub mod mman {
        }
    }

    #[cfg(any(target_os = "linux",
              target_os = "android",
              target_os = "macos",
              target_os = "ios",
              target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "openbsd"))]
    pub mod posix88 {
        pub mod stat_ {
            use types::os::arch::c95::{c_char, c_int};
            use types::os::arch::posix01::stat;
            use types::os::arch::posix88::mode_t;

            extern {
                pub fn chmod(path: *const c_char, mode: mode_t) -> c_int;
                pub fn fchmod(fd: c_int, mode: mode_t) -> c_int;

                #[cfg(any(target_os = "linux",
                          target_os = "freebsd",
                          target_os = "dragonfly",
                          target_os = "openbsd",
                          target_os = "android",
                          target_os = "ios"))]
                pub fn fstat(fildes: c_int, buf: *mut stat) -> c_int;

                #[cfg(target_os = "macos")]
                #[link_name = "fstat64"]
                pub fn fstat(fildes: c_int, buf: *mut stat) -> c_int;

                pub fn mkdir(path: *const c_char, mode: mode_t) -> c_int;
                pub fn mkfifo(path: *const c_char, mode: mode_t) -> c_int;

                #[cfg(any(target_os = "linux",
                          target_os = "freebsd",
                          target_os = "dragonfly",
                          target_os = "openbsd",
                          target_os = "android",
                          target_os = "ios"))]
                pub fn stat(path: *const c_char, buf: *mut stat) -> c_int;

                #[cfg(target_os = "macos")]
                #[link_name = "stat64"]
                pub fn stat(path: *const c_char, buf: *mut stat) -> c_int;
            }
        }

        pub mod stdio {
            use types::common::c95::FILE;
            use types::os::arch::c95::{c_char, c_int};

            extern {
                pub fn popen(command: *const c_char,
                             mode: *const c_char) -> *mut FILE;
                pub fn pclose(stream: *mut FILE) -> c_int;
                pub fn fdopen(fd: c_int, mode: *const c_char) -> *mut FILE;
                pub fn fileno(stream: *mut FILE) -> c_int;
            }
        }

        pub mod fcntl {
            use types::os::arch::c95::{c_char, c_int};
            use types::os::arch::posix88::mode_t;

            extern {
                pub fn open(path: *const c_char, oflag: c_int, mode: mode_t)
                            -> c_int;
                pub fn creat(path: *const c_char, mode: mode_t) -> c_int;
                pub fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
            }
        }

        pub mod dirent {
            use types::common::posix88::{DIR, dirent_t};
            use types::os::arch::c95::{c_char, c_int, c_long};

            // NB: On OS X opendir and readdir have two versions,
            // one for 32-bit kernelspace and one for 64.
            // We should be linking to the 64-bit ones, called
            // opendir$INODE64, etc. but for some reason rustc
            // doesn't link it correctly on i686, so we're going
            // through a C function that mysteriously does work.

            extern {
                #[link_name="rust_opendir"]
                pub fn opendir(dirname: *const c_char) -> *mut DIR;
                #[link_name="rust_readdir_r"]
                pub fn readdir_r(dirp: *mut DIR, entry: *mut dirent_t,
                                  result: *mut *mut dirent_t) -> c_int;
            }

            extern {
                pub fn closedir(dirp: *mut DIR) -> c_int;
                pub fn rewinddir(dirp: *mut DIR);
                pub fn seekdir(dirp: *mut DIR, loc: c_long);
                pub fn telldir(dirp: *mut DIR) -> c_long;
            }
        }

        pub mod unistd {
            use types::common::c95::c_void;
            use types::os::arch::c95::{c_char, c_int, c_long, c_uint};
            use types::os::arch::c95::{size_t};
            use types::os::common::posix01::timespec;
            use types::os::arch::posix01::utimbuf;
            use types::os::arch::posix88::{gid_t, off_t, pid_t};
            use types::os::arch::posix88::{ssize_t, uid_t};

            pub const _PC_NAME_MAX: c_int = 4;

            extern {
                pub fn access(path: *const c_char, amode: c_int) -> c_int;
                pub fn alarm(seconds: c_uint) -> c_uint;
                pub fn chdir(dir: *const c_char) -> c_int;
                pub fn chown(path: *const c_char, uid: uid_t,
                             gid: gid_t) -> c_int;
                pub fn close(fd: c_int) -> c_int;
                pub fn dup(fd: c_int) -> c_int;
                pub fn dup2(src: c_int, dst: c_int) -> c_int;
                pub fn execv(prog: *const c_char,
                             argv: *mut *const c_char) -> c_int;
                pub fn execve(prog: *const c_char, argv: *mut *const c_char,
                              envp: *mut *const c_char)
                              -> c_int;
                pub fn execvp(c: *const c_char,
                              argv: *mut *const c_char) -> c_int;
                pub fn fork() -> pid_t;
                pub fn fpathconf(filedes: c_int, name: c_int) -> c_long;
                pub fn getcwd(buf: *mut c_char, size: size_t) -> *mut c_char;
                pub fn getegid() -> gid_t;
                pub fn geteuid() -> uid_t;
                pub fn getgid() -> gid_t ;
                pub fn getgroups(ngroups_max: c_int, groups: *mut gid_t)
                                 -> c_int;
                pub fn getlogin() -> *mut c_char;
                pub fn getopt(argc: c_int, argv: *mut *const c_char,
                              optstr: *const c_char) -> c_int;
                pub fn getpgrp() -> pid_t;
                pub fn getpid() -> pid_t;
                pub fn getppid() -> pid_t;
                pub fn getuid() -> uid_t;
                pub fn getsid(pid: pid_t) -> pid_t;
                pub fn isatty(fd: c_int) -> c_int;
                pub fn link(src: *const c_char, dst: *const c_char) -> c_int;
                pub fn lseek(fd: c_int, offset: off_t, whence: c_int)
                             -> off_t;
                pub fn pathconf(path: *mut c_char, name: c_int) -> c_long;
                pub fn pause() -> c_int;
                pub fn pipe(fds: *mut c_int) -> c_int;
                pub fn read(fd: c_int, buf: *mut c_void, count: size_t)
                            -> ssize_t;
                pub fn rmdir(path: *const c_char) -> c_int;
                pub fn setgid(gid: gid_t) -> c_int;
                pub fn setpgid(pid: pid_t, pgid: pid_t) -> c_int;
                pub fn setsid() -> pid_t;
                pub fn setuid(uid: uid_t) -> c_int;
                pub fn sleep(secs: c_uint) -> c_uint;
                pub fn usleep(secs: c_uint) -> c_int;
                pub fn nanosleep(rqtp: *const timespec,
                                 rmtp: *mut timespec) -> c_int;
                pub fn sysconf(name: c_int) -> c_long;
                pub fn tcgetpgrp(fd: c_int) -> pid_t;
                pub fn ttyname(fd: c_int) -> *mut c_char;
                pub fn unlink(c: *const c_char) -> c_int;
                pub fn write(fd: c_int, buf: *const c_void, count: size_t)
                             -> ssize_t;
                pub fn pread(fd: c_int, buf: *mut c_void, count: size_t,
                             offset: off_t) -> ssize_t;
                pub fn pwrite(fd: c_int, buf: *const c_void, count: size_t,
                              offset: off_t) -> ssize_t;
                pub fn utime(file: *const c_char, buf: *const utimbuf) -> c_int;
            }
        }

        pub mod signal {
            use types::os::arch::c95::{c_int};
            use types::os::arch::posix88::{pid_t};

            extern {
                pub fn kill(pid: pid_t, sig: c_int) -> c_int;
            }
        }

        pub mod mman {
            use types::common::c95::{c_void};
            use types::os::arch::c95::{size_t, c_int, c_char};
            use types::os::arch::posix88::{mode_t, off_t};

            extern {
                pub fn mlock(addr: *const c_void, len: size_t) -> c_int;
                pub fn munlock(addr: *const c_void, len: size_t) -> c_int;
                pub fn mlockall(flags: c_int) -> c_int;
                pub fn munlockall() -> c_int;

                pub fn mmap(addr: *mut c_void,
                            len: size_t,
                            prot: c_int,
                            flags: c_int,
                            fd: c_int,
                            offset: off_t)
                            -> *mut c_void;
                pub fn munmap(addr: *mut c_void, len: size_t) -> c_int;

                pub fn mprotect(addr: *mut c_void, len: size_t, prot: c_int)
                                -> c_int;

                pub fn msync(addr: *mut c_void, len: size_t, flags: c_int)
                             -> c_int;
                pub fn shm_open(name: *const c_char, oflag: c_int, mode: mode_t)
                                -> c_int;
                pub fn shm_unlink(name: *const c_char) -> c_int;
            }
        }

        pub mod net {
            use types::os::arch::c95::{c_char, c_uint};

            extern {
                pub fn if_nametoindex(ifname: *const c_char) -> c_uint;
            }
        }

    }

    #[cfg(any(target_os = "linux",
              target_os = "android",
              target_os = "macos",
              target_os = "ios",
              target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "openbsd"))]
    pub mod posix01 {
        pub mod stat_ {
            use types::os::arch::c95::{c_char, c_int};
            use types::os::arch::posix01::stat;

            extern {
                #[cfg(any(target_os = "linux",
                          target_os = "freebsd",
                          target_os = "dragonfly",
                          target_os = "openbsd",
                          target_os = "android",
                          target_os = "ios"))]
                pub fn lstat(path: *const c_char, buf: *mut stat) -> c_int;

                #[cfg(target_os = "macos")]
                #[link_name = "lstat64"]
                pub fn lstat(path: *const c_char, buf: *mut stat) -> c_int;
            }
        }

        pub mod unistd {
            use types::os::arch::c95::{c_char, c_int, size_t};
            use types::os::arch::posix88::{ssize_t, off_t};

            extern {
                pub fn readlink(path: *const c_char,
                                buf: *mut c_char,
                                bufsz: size_t)
                                -> ssize_t;

                pub fn fsync(fd: c_int) -> c_int;

                #[cfg(any(target_os = "linux", target_os = "android"))]
                pub fn fdatasync(fd: c_int) -> c_int;

                pub fn setenv(name: *const c_char, val: *const c_char,
                              overwrite: c_int) -> c_int;
                pub fn unsetenv(name: *const c_char) -> c_int;
                pub fn putenv(string: *mut c_char) -> c_int;

                pub fn symlink(path1: *const c_char,
                               path2: *const c_char) -> c_int;

                pub fn ftruncate(fd: c_int, length: off_t) -> c_int;
            }
        }

        pub mod signal {
            use types::os::arch::c95::c_int;
            use types::os::common::posix01::sighandler_t;

            #[cfg(not(all(target_os = "android", target_arch = "arm")))]
            extern {
                pub fn signal(signum: c_int,
                              handler: sighandler_t) -> sighandler_t;
            }

            #[cfg(all(target_os = "android", target_arch = "arm"))]
            extern {
                #[link_name = "bsd_signal"]
                pub fn signal(signum: c_int,
                              handler: sighandler_t) -> sighandler_t;
            }
        }

        pub mod glob {
            use types::os::arch::c95::{c_char, c_int};
            use types::os::common::posix01::{glob_t};

            extern {
                pub fn glob(pattern: *const c_char,
                            flags: c_int,
                            errfunc: ::core::option::Option<extern "C" fn(epath: *const c_char,
                                                              errno: c_int) -> c_int>,
                            pglob: *mut glob_t);
                pub fn globfree(pglob: *mut glob_t);
            }
        }

        pub mod mman {
            use types::common::c95::{c_void};
            use types::os::arch::c95::{c_int, size_t};

            extern {
                pub fn posix_madvise(addr: *mut c_void,
                                     len: size_t,
                                     advice: c_int)
                                     -> c_int;
            }
        }
    }

    #[cfg(target_os = "windows")]
    pub mod posix01 {
        pub mod stat_ {
        }

        pub mod unistd {
        }

        pub mod glob {
        }

        pub mod mman {
        }

        pub mod net {
        }
    }


    #[cfg(any(target_os = "windows",
              target_os = "linux",
              target_os = "android",
              target_os = "macos",
              target_os = "ios",
              target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "openbsd"))]
    pub mod posix08 {
        pub mod unistd {
        }
    }

    #[cfg(not(windows))]
    pub mod bsd43 {
        use types::common::c95::{c_void};
        use types::os::common::bsd44::{socklen_t, sockaddr, ifaddrs};
        use types::os::arch::c95::{c_int, size_t};
        use types::os::arch::posix88::ssize_t;

        extern "system" {
            pub fn socket(domain: c_int, ty: c_int, protocol: c_int) -> c_int;
            pub fn connect(socket: c_int, address: *const sockaddr,
                           len: socklen_t) -> c_int;
            pub fn bind(socket: c_int, address: *const sockaddr,
                        address_len: socklen_t) -> c_int;
            pub fn listen(socket: c_int, backlog: c_int) -> c_int;
            pub fn accept(socket: c_int, address: *mut sockaddr,
                          address_len: *mut socklen_t) -> c_int;
            pub fn getpeername(socket: c_int, address: *mut sockaddr,
                               address_len: *mut socklen_t) -> c_int;
            pub fn getsockname(socket: c_int, address: *mut sockaddr,
                               address_len: *mut socklen_t) -> c_int;
            pub fn setsockopt(socket: c_int, level: c_int, name: c_int,
                              value: *const c_void,
                              option_len: socklen_t) -> c_int;
            pub fn recv(socket: c_int, buf: *mut c_void, len: size_t,
                        flags: c_int) -> ssize_t;
            pub fn send(socket: c_int, buf: *const c_void, len: size_t,
                        flags: c_int) -> ssize_t;
            pub fn recvfrom(socket: c_int, buf: *mut c_void, len: size_t,
                            flags: c_int, addr: *mut sockaddr,
                            addrlen: *mut socklen_t) -> ssize_t;
            pub fn sendto(socket: c_int, buf: *const c_void, len: size_t,
                          flags: c_int, addr: *const sockaddr,
                          addrlen: socklen_t) -> ssize_t;
            pub fn getifaddrs(ifap: *mut *mut ifaddrs) -> c_int;
            pub fn freeifaddrs(ifa: *mut ifaddrs);
            pub fn shutdown(socket: c_int, how: c_int) -> c_int;
        }
    }

    #[cfg(windows)]
    pub mod bsd43 {
        use types::common::c95::{c_void};
        use types::os::common::bsd44::{socklen_t, sockaddr, SOCKET};
        use types::os::arch::c95::c_int;
        use types::os::arch::posix88::ssize_t;

        extern "system" {
            pub fn socket(domain: c_int, ty: c_int, protocol: c_int) -> SOCKET;
            pub fn connect(socket: SOCKET, address: *const sockaddr,
                           len: socklen_t) -> c_int;
            pub fn bind(socket: SOCKET, address: *const sockaddr,
                        address_len: socklen_t) -> c_int;
            pub fn listen(socket: SOCKET, backlog: c_int) -> c_int;
            pub fn accept(socket: SOCKET, address: *mut sockaddr,
                          address_len: *mut socklen_t) -> SOCKET;
            pub fn getpeername(socket: SOCKET, address: *mut sockaddr,
                               address_len: *mut socklen_t) -> c_int;
            pub fn getsockname(socket: SOCKET, address: *mut sockaddr,
                               address_len: *mut socklen_t) -> c_int;
            pub fn setsockopt(socket: SOCKET, level: c_int, name: c_int,
                              value: *const c_void,
                              option_len: socklen_t) -> c_int;
            pub fn closesocket(socket: SOCKET) -> c_int;
            pub fn recv(socket: SOCKET, buf: *mut c_void, len: c_int,
                        flags: c_int) -> c_int;
            pub fn send(socket: SOCKET, buf: *const c_void, len: c_int,
                        flags: c_int) -> c_int;
            pub fn recvfrom(socket: SOCKET, buf: *mut c_void, len: c_int,
                            flags: c_int, addr: *mut sockaddr,
                            addrlen: *mut c_int) -> ssize_t;
            pub fn sendto(socket: SOCKET, buf: *const c_void, len: c_int,
                          flags: c_int, addr: *const sockaddr,
                          addrlen: c_int) -> c_int;
            pub fn shutdown(socket: SOCKET, how: c_int) -> c_int;
        }
    }

    #[cfg(any(target_os = "macos",
              target_os = "ios",
              target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "openbsd"))]
    pub mod bsd44 {
        use types::common::c95::{c_void};
        use types::os::arch::c95::{c_char, c_uchar, c_int, c_uint, c_ulong, size_t};

        extern {
            pub fn ioctl(d: c_int, request: c_ulong, ...) -> c_int;
            pub fn sysctl(name: *mut c_int,
                          namelen: c_uint,
                          oldp: *mut c_void,
                          oldlenp: *mut size_t,
                          newp: *mut c_void,
                          newlen: size_t)
                          -> c_int;
            pub fn sysctlbyname(name: *const c_char,
                                oldp: *mut c_void,
                                oldlenp: *mut size_t,
                                newp: *mut c_void,
                                newlen: size_t)
                                -> c_int;
            pub fn sysctlnametomib(name: *const c_char,
                                   mibp: *mut c_int,
                                   sizep: *mut size_t)
                                   -> c_int;
            pub fn getdtablesize() -> c_int;
            pub fn madvise(addr: *mut c_void, len: size_t, advice: c_int)
                           -> c_int;
            pub fn mincore(addr: *mut c_void, len: size_t, vec: *mut c_uchar)
                           -> c_int;
        }
    }


    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub mod bsd44 {
        use types::common::c95::{c_void};
        use types::os::arch::c95::{c_uchar, c_int, size_t};

        extern {
            #[cfg(not(all(target_os = "android", target_arch = "aarch64")))]
            pub fn getdtablesize() -> c_int;
            pub fn ioctl(d: c_int, request: c_int, ...) -> c_int;
            pub fn madvise(addr: *mut c_void, len: size_t, advice: c_int)
                           -> c_int;
            pub fn mincore(addr: *mut c_void, len: size_t, vec: *mut c_uchar)
                           -> c_int;
        }
    }


    #[cfg(target_os = "windows")]
    pub mod bsd44 {
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub mod extra {
        use types::os::arch::c95::{c_char, c_int};

        extern {
            pub fn _NSGetExecutablePath(buf: *mut c_char, bufsize: *mut u32)
                                        -> c_int;
        }
    }

    #[cfg(any(target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "openbsd"))]
    pub mod extra {
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub mod extra {
    }


    #[cfg(target_os = "windows")]
    pub mod extra {

        pub mod kernel32 {
            use types::os::arch::c95::{c_uint};
            use types::os::arch::extra::{BOOL, DWORD, SIZE_T, HMODULE,
                                               LPCWSTR, LPWSTR,
                                               LPWCH, LPDWORD, LPVOID,
                                               LPCVOID, LPOVERLAPPED,
                                               LPSECURITY_ATTRIBUTES,
                                               LPSTARTUPINFO,
                                               LPPROCESS_INFORMATION,
                                               LPMEMORY_BASIC_INFORMATION,
                                               LPSYSTEM_INFO, HANDLE, LPHANDLE,
                                               LARGE_INTEGER, PLARGE_INTEGER,
                                               LPFILETIME, LPWIN32_FIND_DATAW};

            extern "system" {
                pub fn GetEnvironmentVariableW(n: LPCWSTR,
                                               v: LPWSTR,
                                               nsize: DWORD)
                                               -> DWORD;
                pub fn SetEnvironmentVariableW(n: LPCWSTR, v: LPCWSTR)
                                               -> BOOL;
                pub fn GetEnvironmentStringsW() -> LPWCH;
                pub fn FreeEnvironmentStringsW(env_ptr: LPWCH) -> BOOL;
                pub fn GetModuleFileNameW(hModule: HMODULE,
                                          lpFilename: LPWSTR,
                                          nSize: DWORD)
                                          -> DWORD;
                pub fn CreateDirectoryW(lpPathName: LPCWSTR,
                                        lpSecurityAttributes:
                                        LPSECURITY_ATTRIBUTES)
                                        -> BOOL;
                pub fn CopyFileW(lpExistingFileName: LPCWSTR,
                                 lpNewFileName: LPCWSTR,
                                 bFailIfExists: BOOL)
                                 -> BOOL;
                pub fn DeleteFileW(lpPathName: LPCWSTR) -> BOOL;
                pub fn RemoveDirectoryW(lpPathName: LPCWSTR) -> BOOL;
                pub fn GetCurrentDirectoryW(nBufferLength: DWORD,
                                            lpBuffer: LPWSTR)
                                            -> DWORD;
                pub fn SetCurrentDirectoryW(lpPathName: LPCWSTR) -> BOOL;
                pub fn GetLastError() -> DWORD;
                pub fn FindFirstFileW(fileName: LPCWSTR, findFileData: LPWIN32_FIND_DATAW)
                                      -> HANDLE;
                pub fn FindNextFileW(findFile: HANDLE, findFileData: LPWIN32_FIND_DATAW)
                                     -> BOOL;
                pub fn FindClose(findFile: HANDLE) -> BOOL;
                pub fn DuplicateHandle(hSourceProcessHandle: HANDLE,
                                       hSourceHandle: HANDLE,
                                       hTargetProcessHandle: HANDLE,
                                       lpTargetHandle: LPHANDLE,
                                       dwDesiredAccess: DWORD,
                                       bInheritHandle: BOOL,
                                       dwOptions: DWORD)
                                       -> BOOL;
                pub fn CloseHandle(hObject: HANDLE) -> BOOL;
                pub fn OpenProcess(dwDesiredAccess: DWORD,
                                   bInheritHandle: BOOL,
                                   dwProcessId: DWORD)
                                   -> HANDLE;
                pub fn GetCurrentProcess() -> HANDLE;
                pub fn CreateProcessW(lpApplicationName: LPCWSTR,
                                      lpCommandLine: LPWSTR,
                                      lpProcessAttributes:
                                      LPSECURITY_ATTRIBUTES,
                                      lpThreadAttributes:
                                      LPSECURITY_ATTRIBUTES,
                                      bInheritHandles: BOOL,
                                      dwCreationFlags: DWORD,
                                      lpEnvironment: LPVOID,
                                      lpCurrentDirectory: LPCWSTR,
                                      lpStartupInfo: LPSTARTUPINFO,
                                      lpProcessInformation:
                                      LPPROCESS_INFORMATION)
                                      -> BOOL;
                pub fn WaitForSingleObject(hHandle: HANDLE,
                                           dwMilliseconds: DWORD)
                                           -> DWORD;
                pub fn TerminateProcess(hProcess: HANDLE, uExitCode: c_uint)
                                        -> BOOL;
                pub fn GetExitCodeProcess(hProcess: HANDLE,
                                          lpExitCode: LPDWORD)
                                          -> BOOL;
                pub fn GetSystemInfo(lpSystemInfo: LPSYSTEM_INFO);
                pub fn VirtualAlloc(lpAddress: LPVOID,
                                    dwSize: SIZE_T,
                                    flAllocationType: DWORD,
                                    flProtect: DWORD)
                                    -> LPVOID;
                pub fn VirtualFree(lpAddress: LPVOID,
                                   dwSize: SIZE_T,
                                   dwFreeType: DWORD)
                                   -> BOOL;
                pub fn VirtualLock(lpAddress: LPVOID, dwSize: SIZE_T) -> BOOL;
                pub fn VirtualUnlock(lpAddress: LPVOID, dwSize: SIZE_T)
                                     -> BOOL;
                pub fn VirtualProtect(lpAddress: LPVOID,
                                      dwSize: SIZE_T,
                                      flNewProtect: DWORD,
                                      lpflOldProtect: LPDWORD)
                                      -> BOOL;
                pub fn VirtualQuery(lpAddress: LPCVOID,
                                    lpBuffer: LPMEMORY_BASIC_INFORMATION,
                                    dwLength: SIZE_T)
                                    -> SIZE_T;
                pub fn CreateFileMappingW(hFile: HANDLE,
                                          lpAttributes: LPSECURITY_ATTRIBUTES,
                                          flProtect: DWORD,
                                          dwMaximumSizeHigh: DWORD,
                                          dwMaximumSizeLow: DWORD,
                                          lpName: LPCWSTR)
                                          -> HANDLE;
                pub fn MapViewOfFile(hFileMappingObject: HANDLE,
                                     dwDesiredAccess: DWORD,
                                     dwFileOffsetHigh: DWORD,
                                     dwFileOffsetLow: DWORD,
                                     dwNumberOfBytesToMap: SIZE_T)
                                     -> LPVOID;
                pub fn UnmapViewOfFile(lpBaseAddress: LPCVOID) -> BOOL;
                pub fn MoveFileExW(lpExistingFileName: LPCWSTR,
                                   lpNewFileName: LPCWSTR,
                                   dwFlags: DWORD) -> BOOL;
                pub fn CreateHardLinkW(lpSymlinkFileName: LPCWSTR,
                                       lpTargetFileName: LPCWSTR,
                                       lpSecurityAttributes: LPSECURITY_ATTRIBUTES)
                                        -> BOOL;
                pub fn FlushFileBuffers(hFile: HANDLE) -> BOOL;
                pub fn CreateFileW(lpFileName: LPCWSTR,
                                   dwDesiredAccess: DWORD,
                                   dwShareMode: DWORD,
                                   lpSecurityAttributes: LPSECURITY_ATTRIBUTES,
                                   dwCreationDisposition: DWORD,
                                   dwFlagsAndAttributes: DWORD,
                                   hTemplateFile: HANDLE) -> HANDLE;
                pub fn ReadFile(hFile: HANDLE,
                                lpBuffer: LPVOID,
                                nNumberOfBytesToRead: DWORD,
                                lpNumberOfBytesRead: LPDWORD,
                                lpOverlapped: LPOVERLAPPED) -> BOOL;
                pub fn WriteFile(hFile: HANDLE,
                                 lpBuffer: LPVOID,
                                 nNumberOfBytesToRead: DWORD,
                                 lpNumberOfBytesRead: LPDWORD,
                                 lpOverlapped: LPOVERLAPPED) -> BOOL;
                pub fn SetFilePointerEx(hFile: HANDLE,
                                        liDistanceToMove: LARGE_INTEGER,
                                        lpNewFilePointer: PLARGE_INTEGER,
                                        dwMoveMethod: DWORD) -> BOOL;
                pub fn SetEndOfFile(hFile: HANDLE) -> BOOL;

                pub fn GetSystemTimeAsFileTime(
                            lpSystemTimeAsFileTime: LPFILETIME);

                pub fn QueryPerformanceFrequency(
                            lpFrequency: *mut LARGE_INTEGER) -> BOOL;
                pub fn QueryPerformanceCounter(
                            lpPerformanceCount: *mut LARGE_INTEGER) -> BOOL;

                pub fn GetCurrentProcessId() -> DWORD;
                pub fn CreateNamedPipeW(
                            lpName: LPCWSTR,
                            dwOpenMode: DWORD,
                            dwPipeMode: DWORD,
                            nMaxInstances: DWORD,
                            nOutBufferSize: DWORD,
                            nInBufferSize: DWORD,
                            nDefaultTimeOut: DWORD,
                            lpSecurityAttributes: LPSECURITY_ATTRIBUTES
                            ) -> HANDLE;
                pub fn ConnectNamedPipe(hNamedPipe: HANDLE,
                                        lpOverlapped: LPOVERLAPPED) -> BOOL;
                pub fn WaitNamedPipeW(lpNamedPipeName: LPCWSTR,
                                      nTimeOut: DWORD) -> BOOL;
                pub fn SetNamedPipeHandleState(hNamedPipe: HANDLE,
                                               lpMode: LPDWORD,
                                               lpMaxCollectionCount: LPDWORD,
                                               lpCollectDataTimeout: LPDWORD)
                                                            -> BOOL;
                pub fn CreateEventW(lpEventAttributes: LPSECURITY_ATTRIBUTES,
                                    bManualReset: BOOL,
                                    bInitialState: BOOL,
                                    lpName: LPCWSTR) -> HANDLE;
                pub fn GetOverlappedResult(hFile: HANDLE,
                                           lpOverlapped: LPOVERLAPPED,
                                           lpNumberOfBytesTransferred: LPDWORD,
                                           bWait: BOOL) -> BOOL;
                pub fn DisconnectNamedPipe(hNamedPipe: HANDLE) -> BOOL;
            }
        }

        pub mod msvcrt {
            use types::os::arch::c95::{c_int, c_long};
            use types::os::arch::c99::intptr_t;

            extern {
                #[link_name = "_commit"]
                pub fn commit(fd: c_int) -> c_int;

                #[link_name = "_get_osfhandle"]
                pub fn get_osfhandle(fd: c_int) -> c_long;

                #[link_name = "_open_osfhandle"]
                pub fn open_osfhandle(osfhandle: intptr_t,
                                      flags: c_int) -> c_int;
            }
        }

        pub mod winsock {
            use types::os::arch::c95::{c_int, c_long, c_ulong};
            use types::os::common::bsd44::SOCKET;

            extern "system" {
                pub fn ioctlsocket(s: SOCKET, cmd: c_long, argp: *mut c_ulong) -> c_int;
            }
        }
    }
}

#[doc(hidden)]
pub fn issue_14344_workaround() {} // FIXME #14344 force linkage to happen correctly

#[test] fn work_on_windows() { } // FIXME #10872 needed for a happy windows
