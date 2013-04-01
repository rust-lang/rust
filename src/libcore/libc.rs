// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
* Bindings for the C standard library and other platform libraries
*
* This module contains bindings to the C standard library,
* organized into modules by their defining standard.
* Additionally, it contains some assorted platform-specific definitions.
* For convenience, most functions and types are reexported from `core::libc`,
* so `pub use core::libc::*` will import the available
* C bindings as appropriate for the target platform. The exact
* set of functions available are platform specific.
*
* *Note* Rustdoc does not indicate reexports currently. Also, because these
* definitions are platform-specific, some may not
* appear in the generated documentation.
*
* We consider the following specs reasonably normative with respect
* to interoperating with the C standard library (libc/msvcrt):
*
* * ISO 9899:1990 ('C95', 'ANSI C', 'Standard C'), NA1, 1995.
* * ISO 9899:1999 ('C99' or 'C9x').
* * ISO 9945:1988 / IEEE 1003.1-1988 ('POSIX.1').
* * ISO 9945:2001 / IEEE 1003.1-2001 ('POSIX:2001', 'SUSv3').
* * ISO 9945:2008 / IEEE 1003.1-2008 ('POSIX:2008', 'SUSv4').
*
* Note that any reference to the 1996 revision of POSIX, or any revs
* between 1990 (when '88 was approved at ISO) and 2001 (when the next
* actual revision-revision happened), are merely additions of other
* chapters (1b and 1c) outside the core interfaces.
*
* Despite having several names each, these are *reasonably* coherent
* point-in-time, list-of-definition sorts of specs. You can get each under a
* variety of names but will wind up with the same definition in each case.
*
* See standards(7) in linux-manpages for more details.
*
* Our interface to these libraries is complicated by the non-universality of
* conformance to any of them. About the only thing universally supported is
* the first (C95), beyond that definitions quickly become absent on various
* platforms.
*
* We therefore wind up dividing our module-space up (mostly for the sake of
* sanity while editing, filling-in-details and eliminating duplication) into
* definitions common-to-all (held in modules named c95, c99, posix88, posix01
* and posix08) and definitions that appear only on *some* platforms (named
* 'extra'). This would be things like significant OSX foundation kit, or
* win32 library kernel32.dll, or various fancy glibc, linux or BSD
* extensions.
*
* In addition to the per-platform 'extra' modules, we define a module of
* 'common BSD' libc routines that never quite made it into POSIX but show up
* in multiple derived systems. This is the 4.4BSD r2 / 1995 release, the
* final one from Berkeley after the lawsuits died down and the CSRG
* dissolved.
*/

#[allow(non_camel_case_types)];

// Initial glob-exports mean that all the contents of all the modules
// wind up exported, if you're interested in writing platform-specific code.

pub use libc::types::common::c95::*;
pub use libc::types::common::c99::*;
pub use libc::types::common::posix88::*;
pub use libc::types::common::posix01::*;
pub use libc::types::common::posix08::*;
pub use libc::types::common::bsd44::*;
pub use libc::types::os::common::posix01::*;
pub use libc::types::os::arch::c95::*;
pub use libc::types::os::arch::c99::*;
pub use libc::types::os::arch::posix88::*;
pub use libc::types::os::arch::posix01::*;
pub use libc::types::os::arch::posix08::*;
pub use libc::types::os::arch::bsd44::*;
pub use libc::types::os::arch::extra::*;

pub use libc::consts::os::c95::*;
pub use libc::consts::os::c99::*;
pub use libc::consts::os::posix88::*;
pub use libc::consts::os::posix01::*;
pub use libc::consts::os::posix08::*;
pub use libc::consts::os::bsd44::*;
pub use libc::consts::os::extra::*;

pub use libc::funcs::c95::ctype::*;
pub use libc::funcs::c95::stdio::*;
pub use libc::funcs::c95::stdlib::*;
pub use libc::funcs::c95::string::*;

pub use libc::funcs::posix88::stat_::*;
pub use libc::funcs::posix88::stdio::*;
pub use libc::funcs::posix88::fcntl::*;
pub use libc::funcs::posix88::dirent::*;
pub use libc::funcs::posix88::unistd::*;

pub use libc::funcs::posix01::stat_::*;
pub use libc::funcs::posix01::unistd::*;
pub use libc::funcs::posix08::unistd::*;

pub use libc::funcs::bsd44::*;
pub use libc::funcs::extra::*;

#[cfg(target_os = "win32")]
pub use libc::funcs::extra::kernel32::*;
#[cfg(target_os = "win32")]
pub use libc::funcs::extra::msvcrt::*;

// Explicit export lists for the intersection (provided here) mean that
// you can write more-platform-agnostic code if you stick to just these
// symbols.

pub use libc::types::common::c95::{FILE, c_void, fpos_t};
pub use libc::types::common::posix88::{DIR, dirent_t};
pub use libc::types::os::arch::c95::{c_char, c_double, c_float, c_int};
pub use libc::types::os::arch::c95::{c_long, c_short, c_uchar, c_ulong};
pub use libc::types::os::arch::c95::{c_ushort, clock_t, ptrdiff_t};
pub use libc::types::os::arch::c95::{size_t, time_t};
pub use libc::types::os::arch::c99::{c_longlong, c_ulonglong, intptr_t};
pub use libc::types::os::arch::c99::{uintptr_t};
pub use libc::types::os::arch::posix88::{dev_t, dirent_t, ino_t, mode_t};
pub use libc::types::os::arch::posix88::{off_t, pid_t, ssize_t};

pub use libc::consts::os::c95::{_IOFBF, _IOLBF, _IONBF, BUFSIZ, EOF};
pub use libc::consts::os::c95::{EXIT_FAILURE, EXIT_SUCCESS};
pub use libc::consts::os::c95::{FILENAME_MAX, FOPEN_MAX, L_tmpnam};
pub use libc::consts::os::c95::{RAND_MAX, SEEK_CUR, SEEK_END};
pub use libc::consts::os::c95::{SEEK_SET, TMP_MAX};
pub use libc::consts::os::posix88::{F_OK, O_APPEND, O_CREAT, O_EXCL};
pub use libc::consts::os::posix88::{O_RDONLY, O_RDWR, O_TRUNC, O_WRONLY};
pub use libc::consts::os::posix88::{R_OK, S_IEXEC, S_IFBLK, S_IFCHR};
pub use libc::consts::os::posix88::{S_IFDIR, S_IFIFO, S_IFMT, S_IFREG};
pub use libc::consts::os::posix88::{S_IREAD, S_IRUSR, S_IRWXU, S_IWUSR};
pub use libc::consts::os::posix88::{STDERR_FILENO, STDIN_FILENO};
pub use libc::consts::os::posix88::{STDOUT_FILENO, W_OK, X_OK};

pub use libc::funcs::c95::ctype::{isalnum, isalpha, iscntrl, isdigit};
pub use libc::funcs::c95::ctype::{islower, isprint, ispunct, isspace};
pub use libc::funcs::c95::ctype::{isupper, isxdigit, tolower, toupper};

pub use libc::funcs::c95::stdio::{fclose, feof, ferror, fflush, fgetc};
pub use libc::funcs::c95::stdio::{fgetpos, fgets, fopen, fputc, fputs};
pub use libc::funcs::c95::stdio::{fread, freopen, fseek, fsetpos, ftell};
pub use libc::funcs::c95::stdio::{fwrite, perror, puts, remove, rewind};
pub use libc::funcs::c95::stdio::{setbuf, setvbuf, tmpfile, ungetc};

pub use libc::funcs::c95::stdlib::{abort, abs, atof, atoi, calloc, exit};
pub use libc::funcs::c95::stdlib::{free, getenv, labs, malloc, rand};
pub use libc::funcs::c95::stdlib::{realloc, srand, strtod, strtol};
pub use libc::funcs::c95::stdlib::{strtoul, system};

pub use libc::funcs::c95::string::{memchr, memcmp, memcpy, memmove};
pub use libc::funcs::c95::string::{memset, strcat, strchr, strcmp};
pub use libc::funcs::c95::string::{strcoll, strcpy, strcspn, strerror};
pub use libc::funcs::c95::string::{strlen, strncat, strncmp, strncpy};
pub use libc::funcs::c95::string::{strpbrk, strrchr, strspn, strstr};
pub use libc::funcs::c95::string::{strtok, strxfrm};

pub use libc::funcs::posix88::fcntl::{open, creat};
pub use libc::funcs::posix88::stat_::{chmod, fstat, mkdir, stat};
pub use libc::funcs::posix88::stdio::{fdopen, fileno, pclose, popen};
pub use libc::funcs::posix88::unistd::{access, chdir, close, dup, dup2};
pub use libc::funcs::posix88::unistd::{execv, execve, execvp, getcwd};
pub use libc::funcs::posix88::unistd::{getpid, isatty, lseek, pipe, read};
pub use libc::funcs::posix88::unistd::{rmdir, unlink, write};


pub mod types {

    // Types tend to vary *per architecture* so we pull their definitions out
    // into this module.

    // Standard types that are opaque or common, so are not per-target.
    pub mod common {
        pub mod c95 {
            pub enum c_void {}
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

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    pub mod os {
        pub mod common {
            pub mod posix01 {}
        }

        #[cfg(target_arch = "x86")]
        #[cfg(target_arch = "arm")]
        #[cfg(target_arch = "mips")]
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
                pub type wchar_t = i32;
            }
            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;
                pub type intptr_t = int;
                pub type uintptr_t = uint;
            }
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
            pub mod posix01 {
                use libc::types::os::arch::c95::{c_int, c_short, c_long,
                                                 time_t};
                use libc::types::os::arch::posix88::{dev_t, gid_t, ino_t};
                use libc::types::os::arch::posix88::{mode_t, off_t};
                use libc::types::os::arch::posix88::{uid_t};

                pub type nlink_t = u32;
                pub type blksize_t = i32;
                pub type blkcnt_t = i32;
                pub struct stat {
                    st_dev: dev_t,
                    __pad1: c_short,
                    st_ino: ino_t,
                    st_mode: mode_t,
                    st_nlink: nlink_t,
                    st_uid: uid_t,
                    st_gid: gid_t,
                    st_rdev: dev_t,
                    __pad2: c_short,
                    st_size: off_t,
                    st_blksize: blksize_t,
                    st_blocks: blkcnt_t,
                    st_atime: time_t,
                    st_atime_nsec: c_long,
                    st_mtime: time_t,
                    st_mtime_nsec: c_long,
                    st_ctime: time_t,
                    st_ctime_nsec: c_long,
                    __unused4: c_long,
                    __unused5: c_long,
                }
            }
            pub mod posix08 {}
            pub mod bsd44 {}
            pub mod extra {}
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
                pub type wchar_t = i32;
            }
            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;
                pub type intptr_t = int;
                pub type uintptr_t = uint;
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
            pub mod posix01 {
                use libc::types::os::arch::c95::{c_int, c_long, time_t};
                use libc::types::os::arch::posix88::{dev_t, gid_t, ino_t};
                use libc::types::os::arch::posix88::{mode_t, off_t};
                use libc::types::os::arch::posix88::{uid_t};

                pub type nlink_t = u64;
                pub type blksize_t = i64;
                pub type blkcnt_t = i64;
                pub struct stat {
                    st_dev: dev_t,
                    st_ino: ino_t,
                    st_nlink: nlink_t,
                    st_mode: mode_t,
                    st_uid: uid_t,
                    st_gid: gid_t,
                    __pad0: c_int,
                    st_rdev: dev_t,
                    st_size: off_t,
                    st_blksize: blksize_t,
                    st_blocks: blkcnt_t,
                    st_atime: time_t,
                    st_atime_nsec: c_long,
                    st_mtime: time_t,
                    st_mtime_nsec: c_long,
                    st_ctime: time_t,
                    st_ctime_nsec: c_long,
                    __unused: [c_long, ..3],
                }
            }
            pub mod posix08 {
            }
            pub mod bsd44 {
            }
            pub mod extra {
            }
        }
    }

    #[cfg(target_os = "freebsd")]
    pub mod os {
        pub mod common {
            pub mod posix01 {}
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
                pub type wchar_t = i32;
            }
            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;
                pub type intptr_t = int;
                pub type uintptr_t = uint;
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
                use libc::types::common::c99::{uint8_t, uint32_t, int32_t};
                use libc::types::os::arch::c95::{c_long, time_t};
                use libc::types::os::arch::posix88::{dev_t, gid_t, ino_t};
                use libc::types::os::arch::posix88::{mode_t, off_t};
                use libc::types::os::arch::posix88::{uid_t};

                pub type nlink_t = u16;
                pub type blksize_t = i64;
                pub type blkcnt_t = i64;
                pub type fflags_t = u32;
                pub struct stat {
                    st_dev: dev_t,
                    st_ino: ino_t,
                    st_mode: mode_t,
                    st_nlink: nlink_t,
                    st_uid: uid_t,
                    st_gid: gid_t,
                    st_rdev: dev_t,
                    st_atime: time_t,
                    st_atime_nsec: c_long,
                    st_mtime: time_t,
                    st_mtime_nsec: c_long,
                    st_ctime: time_t,
                    st_ctime_nsec: c_long,
                    st_size: off_t,
                    st_blocks: blkcnt_t,
                    st_blksize: blksize_t,
                    st_flags: fflags_t,
                    st_gen: uint32_t,
                    st_lspare: int32_t,
                    st_birthtime: time_t,
                    st_birthtime_nsec: c_long,
                    __unused: [uint8_t, ..2],
                }
            }
            pub mod posix08 {
            }
            pub mod bsd44 {
            }
            pub mod extra {
            }
        }
    }

    #[cfg(target_os = "win32")]
    pub mod os {
        pub mod common {
            pub mod posix01 {
                use libc::types::os::arch::c95::{c_int, c_short};
                use libc::types::os::arch::extra::{int64, time64_t};
                use libc::types::os::arch::posix88::{dev_t, ino_t};
                use libc::types::os::arch::posix88::mode_t;

                // Note: this is the struct called stat64 in win32. Not stat,
                // nor stati64.
                pub struct stat {
                    st_dev: dev_t,
                    st_ino: ino_t,
                    st_mode: mode_t,
                    st_nlink: c_short,
                    st_uid: c_short,
                    st_gid: c_short,
                    st_rdev: dev_t,
                    st_size: int64,
                    st_atime: time64_t,
                    st_mtime: time64_t,
                    st_ctime: time64_t,
                }
            }
        }

        #[cfg(target_arch = "x86")]
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
                pub type wchar_t = u16;
            }
            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;
                pub type intptr_t = int;
                pub type uintptr_t = uint;
            }
            pub mod posix88 {
                pub type off_t = i32;
                pub type dev_t = u32;
                pub type ino_t = i16;
                pub type pid_t = i32;
                pub type useconds_t = u32;
                pub type mode_t = u16;
                pub type ssize_t = i32;
            }
            pub mod posix01 {
            }
            pub mod posix08 {
            }
            pub mod bsd44 {
            }
            pub mod extra {
                use libc::types::common::c95::c_void;
                use libc::types::os::arch::c95::{c_char, c_int, c_uint};
                use libc::types::os::arch::c95::{c_long, c_ulong};
                use libc::types::os::arch::c95::{wchar_t};
                use libc::types::os::arch::c99::{c_ulonglong};

                pub type BOOL = c_int;
                pub type BYTE = u8;
                pub type CCHAR = c_char;
                pub type CHAR = c_char;

                pub type DWORD = c_ulong;
                pub type DWORDLONG = c_ulonglong;

                pub type HANDLE = LPVOID;
                pub type HMODULE = c_uint;

                pub type LONG_PTR = c_long;

                pub type LPCWSTR = *WCHAR;
                pub type LPCSTR = *CHAR;
                pub type LPCTSTR = *CHAR;
                pub type LPTCH = *CHAR;

                pub type LPWSTR = *mut WCHAR;
                pub type LPSTR = *mut CHAR;

                // Not really, but opaque to us.
                pub type LPSECURITY_ATTRIBUTES = LPVOID;

                pub type LPVOID = *mut c_void;
                pub type LPWORD = *mut WORD;

                pub type LRESULT = LONG_PTR;
                pub type PBOOL = *mut BOOL;
                pub type WCHAR = wchar_t;
                pub type WORD = u16;

                pub type time64_t = i64;
                pub type int64 = i64;
            }
        }
    }

    #[cfg(target_os = "macos")]
    pub mod os {
        pub mod common {
            pub mod posix01 {
            }
        }

        #[cfg(target_arch = "x86")]
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
                pub type clock_t = u32;
                pub type time_t = i32;
                pub type wchar_t = i32;
            }
            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;
                pub type intptr_t = int;
                pub type uintptr_t = uint;
            }
            pub mod posix88 {
                pub type off_t = i64;
                pub type dev_t = i32;
                pub type ino_t = u64;
                pub type pid_t = i32;
                pub type uid_t = u32;
                pub type gid_t = u32;
                pub type useconds_t = u32;
                pub type mode_t = u16;
                pub type ssize_t = i32;
            }
            pub mod posix01 {
                use libc::types::common::c99::{int32_t, int64_t, uint32_t};
                use libc::types::os::arch::c95::{c_long, time_t};
                use libc::types::os::arch::posix88::{dev_t, gid_t, ino_t,
                                                     mode_t, off_t, uid_t};

                pub type nlink_t = u16;
                pub type blksize_t = i64;
                pub type blkcnt_t = i32;

                pub struct stat {
                    st_dev: dev_t,
                    st_mode: mode_t,
                    st_nlink: nlink_t,
                    st_ino: ino_t,
                    st_uid: uid_t,
                    st_gid: gid_t,
                    st_rdev: dev_t,
                    st_atime: time_t,
                    st_atime_nsec: c_long,
                    st_mtime: time_t,
                    st_mtime_nsec: c_long,
                    st_ctime: time_t,
                    st_ctime_nsec: c_long,
                    st_birthtime: time_t,
                    st_birthtime_nsec: c_long,
                    st_size: off_t,
                    st_blocks: blkcnt_t,
                    st_blksize: blksize_t,
                    st_flags: uint32_t,
                    st_gen: uint32_t,
                    st_lspare: int32_t,
                    st_qspare: [int64_t, ..2],
                }
            }
            pub mod posix08 {
            }
            pub mod bsd44 {
            }
            pub mod extra {
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
                pub type clock_t = u64;
                pub type time_t = i64;
                pub type wchar_t = i32;
            }
            pub mod c99 {
                pub type c_longlong = i64;
                pub type c_ulonglong = u64;
                pub type intptr_t = int;
                pub type uintptr_t = uint;
            }
            pub mod posix88 {
                pub type off_t = i64;
                pub type dev_t = i32;
                pub type ino_t = u64;
                pub type pid_t = i32;
                pub type uid_t = u32;
                pub type gid_t = u32;
                pub type useconds_t = u32;
                pub type mode_t = u16;
                pub type ssize_t = i64;
            }
            pub mod posix01 {
                use libc::types::common::c99::{int32_t, int64_t};
                use libc::types::common::c99::{uint32_t};
                use libc::types::os::arch::c95::{c_long, time_t};
                use libc::types::os::arch::posix88::{dev_t, gid_t, ino_t};
                use libc::types::os::arch::posix88::{mode_t, off_t, uid_t};

                pub type nlink_t = u16;
                pub type blksize_t = i64;
                pub type blkcnt_t = i32;

                pub struct stat {
                    st_dev: dev_t,
                    st_mode: mode_t,
                    st_nlink: nlink_t,
                    st_ino: ino_t,
                    st_uid: uid_t,
                    st_gid: gid_t,
                    st_rdev: dev_t,
                    st_atime: time_t,
                    st_atime_nsec: c_long,
                    st_mtime: time_t,
                    st_mtime_nsec: c_long,
                    st_ctime: time_t,
                    st_ctime_nsec: c_long,
                    st_birthtime: time_t,
                    st_birthtime_nsec: c_long,
                    st_size: off_t,
                    st_blocks: blkcnt_t,
                    st_blksize: blksize_t,
                    st_flags: uint32_t,
                    st_gen: uint32_t,
                    st_lspare: int32_t,
                    st_qspare: [int64_t, ..2],
                }
            }
            pub mod posix08 {
            }
            pub mod bsd44 {
            }
            pub mod extra {
            }
        }
    }
}

pub mod consts {
    // Consts tend to vary per OS so we pull their definitions out
    // into this module.

    #[cfg(target_os = "win32")]
    pub mod os {
        pub mod c95 {
            pub static EXIT_FAILURE : int = 1;
            pub static EXIT_SUCCESS : int = 0;
            pub static RAND_MAX : int = 32767;
            pub static EOF : int = -1;
            pub static SEEK_SET : int = 0;
            pub static SEEK_CUR : int = 1;
            pub static SEEK_END : int = 2;
            pub static _IOFBF : int = 0;
            pub static _IONBF : int = 4;
            pub static _IOLBF : int = 64;
            pub static BUFSIZ : uint = 512_u;
            pub static FOPEN_MAX : uint = 20_u;
            pub static FILENAME_MAX : uint = 260_u;
            pub static L_tmpnam : uint = 16_u;
            pub static TMP_MAX : uint = 32767_u;
        }
        pub mod c99 {
        }
        pub mod posix88 {
            pub static O_RDONLY : int = 0;
            pub static O_WRONLY : int = 1;
            pub static O_RDWR : int = 2;
            pub static O_APPEND : int = 8;
            pub static O_CREAT : int = 256;
            pub static O_EXCL : int = 1024;
            pub static O_TRUNC : int = 512;
            pub static S_IFIFO : int = 4096;
            pub static S_IFCHR : int = 8192;
            pub static S_IFBLK : int = 12288;
            pub static S_IFDIR : int = 16384;
            pub static S_IFREG : int = 32768;
            pub static S_IFMT : int = 61440;
            pub static S_IEXEC : int = 64;
            pub static S_IWRITE : int = 128;
            pub static S_IREAD : int = 256;
            pub static S_IRWXU : int = 448;
            pub static S_IXUSR : int = 64;
            pub static S_IWUSR : int = 128;
            pub static S_IRUSR : int = 256;
            pub static F_OK : int = 0;
            pub static R_OK : int = 4;
            pub static W_OK : int = 2;
            pub static X_OK : int = 1;
            pub static STDIN_FILENO : int = 0;
            pub static STDOUT_FILENO : int = 1;
            pub static STDERR_FILENO : int = 2;
        }
        pub mod posix01 {
        }
        pub mod posix08 {
        }
        pub mod bsd44 {
        }
        pub mod extra {
            pub static O_TEXT : int = 16384;
            pub static O_BINARY : int = 32768;
            pub static O_NOINHERIT: int = 128;

            pub static ERROR_SUCCESS : int = 0;
            pub static ERROR_INSUFFICIENT_BUFFER : int = 122;
            pub static INVALID_HANDLE_VALUE: int = -1;
        }
    }


    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    pub mod os {
        pub mod c95 {
            pub static EXIT_FAILURE : int = 1;
            pub static EXIT_SUCCESS : int = 0;
            pub static RAND_MAX : int = 2147483647;
            pub static EOF : int = -1;
            pub static SEEK_SET : int = 0;
            pub static SEEK_CUR : int = 1;
            pub static SEEK_END : int = 2;
            pub static _IOFBF : int = 0;
            pub static _IONBF : int = 2;
            pub static _IOLBF : int = 1;
            pub static BUFSIZ : uint = 8192_u;
            pub static FOPEN_MAX : uint = 16_u;
            pub static FILENAME_MAX : uint = 4096_u;
            pub static L_tmpnam : uint = 20_u;
            pub static TMP_MAX : uint = 238328_u;
        }
        pub mod c99 {
        }
        pub mod posix88 {
            pub static O_RDONLY : int = 0;
            pub static O_WRONLY : int = 1;
            pub static O_RDWR : int = 2;
            pub static O_APPEND : int = 1024;
            pub static O_CREAT : int = 64;
            pub static O_EXCL : int = 128;
            pub static O_TRUNC : int = 512;
            pub static S_IFIFO : int = 4096;
            pub static S_IFCHR : int = 8192;
            pub static S_IFBLK : int = 24576;
            pub static S_IFDIR : int = 16384;
            pub static S_IFREG : int = 32768;
            pub static S_IFMT : int = 61440;
            pub static S_IEXEC : int = 64;
            pub static S_IWRITE : int = 128;
            pub static S_IREAD : int = 256;
            pub static S_IRWXU : int = 448;
            pub static S_IXUSR : int = 64;
            pub static S_IWUSR : int = 128;
            pub static S_IRUSR : int = 256;
            pub static F_OK : int = 0;
            pub static R_OK : int = 4;
            pub static W_OK : int = 2;
            pub static X_OK : int = 1;
            pub static STDIN_FILENO : int = 0;
            pub static STDOUT_FILENO : int = 1;
            pub static STDERR_FILENO : int = 2;
            pub static F_LOCK : int = 1;
            pub static F_TEST : int = 3;
            pub static F_TLOCK : int = 2;
            pub static F_ULOCK : int = 0;
        }
        pub mod posix01 {
        }
        pub mod posix08 {
        }
        pub mod bsd44 {
        }
        pub mod extra {
            pub static O_RSYNC : int = 1052672;
            pub static O_DSYNC : int = 4096;
            pub static O_SYNC : int = 1052672;
        }
    }

    #[cfg(target_os = "freebsd")]
    pub mod os {
        pub mod c95 {
            pub static EXIT_FAILURE : int = 1;
            pub static EXIT_SUCCESS : int = 0;
            pub static RAND_MAX : int = 2147483647;
            pub static EOF : int = -1;
            pub static SEEK_SET : int = 0;
            pub static SEEK_CUR : int = 1;
            pub static SEEK_END : int = 2;
            pub static _IOFBF : int = 0;
            pub static _IONBF : int = 2;
            pub static _IOLBF : int = 1;
            pub static BUFSIZ : uint = 1024_u;
            pub static FOPEN_MAX : uint = 20_u;
            pub static FILENAME_MAX : uint = 1024_u;
            pub static L_tmpnam : uint = 1024_u;
            pub static TMP_MAX : uint = 308915776_u;
        }
        pub mod c99 {
        }
        pub mod posix88 {
            pub static O_RDONLY : int = 0;
            pub static O_WRONLY : int = 1;
            pub static O_RDWR : int = 2;
            pub static O_APPEND : int = 8;
            pub static O_CREAT : int = 512;
            pub static O_EXCL : int = 2048;
            pub static O_TRUNC : int = 1024;
            pub static S_IFIFO : int = 4096;
            pub static S_IFCHR : int = 8192;
            pub static S_IFBLK : int = 24576;
            pub static S_IFDIR : int = 16384;
            pub static S_IFREG : int = 32768;
            pub static S_IFMT : int = 61440;
            pub static S_IEXEC : int = 64;
            pub static S_IWRITE : int = 128;
            pub static S_IREAD : int = 256;
            pub static S_IRWXU : int = 448;
            pub static S_IXUSR : int = 64;
            pub static S_IWUSR : int = 128;
            pub static S_IRUSR : int = 256;
            pub static F_OK : int = 0;
            pub static R_OK : int = 4;
            pub static W_OK : int = 2;
            pub static X_OK : int = 1;
            pub static STDIN_FILENO : int = 0;
            pub static STDOUT_FILENO : int = 1;
            pub static STDERR_FILENO : int = 2;
            pub static F_LOCK : int = 1;
            pub static F_TEST : int = 3;
            pub static F_TLOCK : int = 2;
            pub static F_ULOCK : int = 0;
        }
        pub mod posix01 {
        }
        pub mod posix08 {
        }
        pub mod bsd44 {
        }
        pub mod extra {
            pub static O_SYNC : int = 128;
            pub static CTL_KERN: int = 1;
            pub static KERN_PROC: int = 14;
            pub static KERN_PROC_PATHNAME: int = 12;
        }
    }

    #[cfg(target_os = "macos")]
    pub mod os {
        pub mod c95 {
            pub static EXIT_FAILURE : int = 1;
            pub static EXIT_SUCCESS : int = 0;
            pub static RAND_MAX : int = 2147483647;
            pub static EOF : int = -1;
            pub static SEEK_SET : int = 0;
            pub static SEEK_CUR : int = 1;
            pub static SEEK_END : int = 2;
            pub static _IOFBF : int = 0;
            pub static _IONBF : int = 2;
            pub static _IOLBF : int = 1;
            pub static BUFSIZ : uint = 1024_u;
            pub static FOPEN_MAX : uint = 20_u;
            pub static FILENAME_MAX : uint = 1024_u;
            pub static L_tmpnam : uint = 1024_u;
            pub static TMP_MAX : uint = 308915776_u;
        }
        pub mod c99 {
        }
        pub mod posix88 {
            pub static O_RDONLY : int = 0;
            pub static O_WRONLY : int = 1;
            pub static O_RDWR : int = 2;
            pub static O_APPEND : int = 8;
            pub static O_CREAT : int = 512;
            pub static O_EXCL : int = 2048;
            pub static O_TRUNC : int = 1024;
            pub static S_IFIFO : int = 4096;
            pub static S_IFCHR : int = 8192;
            pub static S_IFBLK : int = 24576;
            pub static S_IFDIR : int = 16384;
            pub static S_IFREG : int = 32768;
            pub static S_IFMT : int = 61440;
            pub static S_IEXEC : int = 64;
            pub static S_IWRITE : int = 128;
            pub static S_IREAD : int = 256;
            pub static S_IRWXU : int = 448;
            pub static S_IXUSR : int = 64;
            pub static S_IWUSR : int = 128;
            pub static S_IRUSR : int = 256;
            pub static F_OK : int = 0;
            pub static R_OK : int = 4;
            pub static W_OK : int = 2;
            pub static X_OK : int = 1;
            pub static STDIN_FILENO : int = 0;
            pub static STDOUT_FILENO : int = 1;
            pub static STDERR_FILENO : int = 2;
            pub static F_LOCK : int = 1;
            pub static F_TEST : int = 3;
            pub static F_TLOCK : int = 2;
            pub static F_ULOCK : int = 0;
        }
        pub mod posix01 {
        }
        pub mod posix08 {
        }
        pub mod bsd44 {
        }
        pub mod extra {
            pub static O_DSYNC : int = 4194304;
            pub static O_SYNC : int = 128;
            pub static F_FULLFSYNC : int = 51;
        }
    }
}


pub mod funcs {
    // Thankfull most of c95 is universally available and does not vary by OS
    // or anything. The same is not true of POSIX.

    pub mod c95 {
        #[nolink]
        #[abi = "cdecl"]
        pub mod ctype {
            use libc::types::os::arch::c95::{c_char, c_int};

            pub extern {
                unsafe fn isalnum(c: c_int) -> c_int;
                unsafe fn isalpha(c: c_int) -> c_int;
                unsafe fn iscntrl(c: c_int) -> c_int;
                unsafe fn isdigit(c: c_int) -> c_int;
                unsafe fn isgraph(c: c_int) -> c_int;
                unsafe fn islower(c: c_int) -> c_int;
                unsafe fn isprint(c: c_int) -> c_int;
                unsafe fn ispunct(c: c_int) -> c_int;
                unsafe fn isspace(c: c_int) -> c_int;
                unsafe fn isupper(c: c_int) -> c_int;
                unsafe fn isxdigit(c: c_int) -> c_int;
                unsafe fn tolower(c: c_char) -> c_char;
                unsafe fn toupper(c: c_char) -> c_char;
            }
        }

        #[nolink]
        #[abi = "cdecl"]
        pub mod stdio {
            use libc::types::common::c95::{FILE, c_void, fpos_t};
            use libc::types::os::arch::c95::{c_char, c_int, c_long, size_t};

            pub extern {
                unsafe fn fopen(filename: *c_char, mode: *c_char) -> *FILE;
                unsafe fn freopen(filename: *c_char, mode: *c_char,
                           file: *FILE) -> *FILE;
                unsafe fn fflush(file: *FILE) -> c_int;
                unsafe fn fclose(file: *FILE) -> c_int;
                unsafe fn remove(filename: *c_char) -> c_int;
                unsafe fn rename(oldname: *c_char, newname: *c_char) -> c_int;
                unsafe fn tmpfile() -> *FILE;
                unsafe fn setvbuf(stream: *FILE, buffer: *c_char,
                           mode: c_int, size: size_t) -> c_int;
                unsafe fn setbuf(stream: *FILE, buf: *c_char);
                // Omitted: printf and scanf variants.
                unsafe fn fgetc(stream: *FILE) -> c_int;
                unsafe fn fgets(buf: *mut c_char, n: c_int,
                         stream: *FILE) -> *c_char;
                unsafe fn fputc(c: c_int, stream: *FILE) -> c_int;
                unsafe fn fputs(s: *c_char, stream: *FILE) -> *c_char;
                // Omitted: getc, getchar (might be macros).

                // Omitted: gets, so ridiculously unsafe that it should not
                // survive.

                // Omitted: putc, putchar (might be macros).
                unsafe fn puts(s: *c_char) -> c_int;
                unsafe fn ungetc(c: c_int, stream: *FILE) -> c_int;
                unsafe fn fread(ptr: *mut c_void, size: size_t,
                         nobj: size_t, stream: *FILE) -> size_t;
                unsafe fn fwrite(ptr: *c_void, size: size_t,
                          nobj: size_t, stream: *FILE) -> size_t;
                unsafe fn fseek(stream: *FILE, offset: c_long,
                         whence: c_int) -> c_int;
                unsafe fn ftell(stream: *FILE) -> c_long;
                unsafe fn rewind(stream: *FILE);
                unsafe fn fgetpos(stream: *FILE, ptr: *fpos_t) -> c_int;
                unsafe fn fsetpos(stream: *FILE, ptr: *fpos_t) -> c_int;
                unsafe fn feof(stream: *FILE) -> c_int;
                unsafe fn ferror(stream: *FILE) -> c_int;
                unsafe fn perror(s: *c_char);
            }
        }

        #[nolink]
        #[abi = "cdecl"]
        pub mod stdlib {
            use libc::types::common::c95::c_void;
            use libc::types::os::arch::c95::{c_char, c_double, c_int};
            use libc::types::os::arch::c95::{c_long, c_uint, c_ulong};
            use libc::types::os::arch::c95::{size_t};

            pub extern {
                unsafe fn abs(i: c_int) -> c_int;
                unsafe fn labs(i: c_long) -> c_long;
                // Omitted: div, ldiv (return pub type incomplete).
                unsafe fn atof(s: *c_char) -> c_double;
                unsafe fn atoi(s: *c_char) -> c_int;
                unsafe fn strtod(s: *c_char, endp: **c_char) -> c_double;
                unsafe fn strtol(s: *c_char, endp: **c_char, base: c_int)
                              -> c_long;
                unsafe fn strtoul(s: *c_char, endp: **c_char, base: c_int)
                               -> c_ulong;
                unsafe fn calloc(nobj: size_t, size: size_t) -> *c_void;
                unsafe fn malloc(size: size_t) -> *c_void;
                unsafe fn realloc(p: *c_void, size: size_t) -> *c_void;
                unsafe fn free(p: *c_void);
                unsafe fn abort() -> !;
                unsafe fn exit(status: c_int) -> !;
                // Omitted: atexit.
                unsafe fn system(s: *c_char) -> c_int;
                unsafe fn getenv(s: *c_char) -> *c_char;
                // Omitted: bsearch, qsort
                unsafe fn rand() -> c_int;
                unsafe fn srand(seed: c_uint);
            }
        }

        #[nolink]
        #[abi = "cdecl"]
        pub mod string {
            use libc::types::common::c95::c_void;
            use libc::types::os::arch::c95::{c_char, c_int, size_t};
            use libc::types::os::arch::c95::{wchar_t};

            pub extern {
                unsafe fn strcpy(dst: *c_char, src: *c_char) -> *c_char;
                unsafe fn strncpy(dst: *c_char, src: *c_char, n: size_t)
                               -> *c_char;
                unsafe fn strcat(s: *c_char, ct: *c_char) -> *c_char;
                unsafe fn strncat(s: *c_char, ct: *c_char, n: size_t)
                               -> *c_char;
                unsafe fn strcmp(cs: *c_char, ct: *c_char) -> c_int;
                unsafe fn strncmp(cs: *c_char, ct: *c_char, n: size_t)
                               -> c_int;
                unsafe fn strcoll(cs: *c_char, ct: *c_char) -> c_int;
                unsafe fn strchr(cs: *c_char, c: c_int) -> *c_char;
                unsafe fn strrchr(cs: *c_char, c: c_int) -> *c_char;
                unsafe fn strspn(cs: *c_char, ct: *c_char) -> size_t;
                unsafe fn strcspn(cs: *c_char, ct: *c_char) -> size_t;
                unsafe fn strpbrk(cs: *c_char, ct: *c_char) -> *c_char;
                unsafe fn strstr(cs: *c_char, ct: *c_char) -> *c_char;
                unsafe fn strlen(cs: *c_char) -> size_t;
                unsafe fn strerror(n: c_int) -> *c_char;
                unsafe fn strtok(s: *c_char, t: *c_char) -> *c_char;
                unsafe fn strxfrm(s: *c_char, ct: *c_char, n: size_t)
                               -> size_t;
                unsafe fn wcslen(buf: *wchar_t) -> size_t;

                // These are fine to execute on the Rust stack. They must be,
                // in fact, because LLVM generates calls to them!
                #[rust_stack]
                #[inline(always)]
                unsafe fn memcpy(s: *c_void, ct: *c_void, n: size_t)
                              -> *c_void;
                #[rust_stack]
                #[inline(always)]
                unsafe fn memmove(s: *c_void, ct: *c_void, n: size_t)
                               -> *c_void;
                #[rust_stack]
                #[inline(always)]
                unsafe fn memcmp(cx: *c_void, ct: *c_void, n: size_t)
                              -> c_int;
                #[rust_stack]
                #[inline(always)]
                unsafe fn memchr(cx: *c_void, c: c_int, n: size_t) -> *c_void;
                #[rust_stack]
                #[inline(always)]
                unsafe fn memset(s: *c_void, c: c_int, n: size_t) -> *c_void;
            }
        }
    }

    // Microsoft helpfully underscore-qualifies all of its POSIX-like symbols
    // to make sure you don't use them accidentally. It also randomly deviates
    // from the exact signatures you might otherwise expect, and omits much,
    // so be careful when trying to write portable code; it won't always work
    // with the same POSIX functions and types as other platforms.

    #[cfg(target_os = "win32")]
    pub mod posix88 {
        #[nolink]
        #[abi = "cdecl"]
        pub mod stat_ {
            use libc::types::os::common::posix01::stat;
            use libc::types::os::arch::c95::{c_int, c_char};

            pub extern {
                #[link_name = "_chmod"]
                unsafe fn chmod(path: *c_char, mode: c_int) -> c_int;

                #[link_name = "_mkdir"]
                unsafe fn mkdir(path: *c_char) -> c_int;

                #[link_name = "_fstat64"]
                unsafe fn fstat(fildes: c_int, buf: *mut stat) -> c_int;

                #[link_name = "_stat64"]
                unsafe fn stat(path: *c_char, buf: *mut stat) -> c_int;
            }
        }

        #[nolink]
        #[abi = "cdecl"]
        pub mod stdio {
            use libc::types::common::c95::FILE;
            use libc::types::os::arch::c95::{c_int, c_char};

            pub extern {
                #[link_name = "_popen"]
                unsafe fn popen(command: *c_char, mode: *c_char) -> *FILE;

                #[link_name = "_pclose"]
                unsafe fn pclose(stream: *FILE) -> c_int;

                #[link_name = "_fdopen"]
                unsafe fn fdopen(fd: c_int, mode: *c_char) -> *FILE;

                #[link_name = "_fileno"]
                unsafe fn fileno(stream: *FILE) -> c_int;
            }
        }

        #[nolink]
        #[abi = "cdecl"]
        pub mod fcntl {
            use libc::types::os::arch::c95::{c_int, c_char};
            pub extern {
                #[link_name = "_open"]
                unsafe fn open(path: *c_char, oflag: c_int, mode: c_int)
                            -> c_int;

                #[link_name = "_creat"]
                unsafe fn creat(path: *c_char, mode: c_int) -> c_int;
            }
        }

        #[nolink]
        #[abi = "cdecl"]
        pub mod dirent {
            // Not supplied at all.
        }

        #[nolink]
        #[abi = "cdecl"]
        pub mod unistd {
            use libc::types::common::c95::c_void;
            use libc::types::os::arch::c95::{c_int, c_uint, c_char,
                                             c_long, size_t};
            use libc::types::os::arch::c99::intptr_t;

            pub extern {
                #[link_name = "_access"]
                unsafe fn access(path: *c_char, amode: c_int) -> c_int;

                #[link_name = "_chdir"]
                unsafe fn chdir(dir: *c_char) -> c_int;

                #[link_name = "_close"]
                unsafe fn close(fd: c_int) -> c_int;

                #[link_name = "_dup"]
                unsafe fn dup(fd: c_int) -> c_int;

                #[link_name = "_dup2"]
                unsafe fn dup2(src: c_int, dst: c_int) -> c_int;

                #[link_name = "_execv"]
                unsafe fn execv(prog: *c_char, argv: **c_char) -> intptr_t;

                #[link_name = "_execve"]
                unsafe fn execve(prog: *c_char, argv: **c_char,
                          envp: **c_char) -> c_int;

                #[link_name = "_execvp"]
                unsafe fn execvp(c: *c_char, argv: **c_char) -> c_int;

                #[link_name = "_execvpe"]
                unsafe fn execvpe(c: *c_char, argv: **c_char,
                           envp: **c_char) -> c_int;

                #[link_name = "_getcwd"]
                unsafe fn getcwd(buf: *c_char, size: size_t) -> *c_char;

                #[link_name = "_getpid"]
                unsafe fn getpid() -> c_int;

                #[link_name = "_isatty"]
                unsafe fn isatty(fd: c_int) -> c_int;

                #[link_name = "_lseek"]
                unsafe fn lseek(fd: c_int, offset: c_long, origin: c_int)
                             -> c_long;

                #[link_name = "_pipe"]
                unsafe fn pipe(fds: *mut c_int, psize: c_uint,
                        textmode: c_int) -> c_int;

                #[link_name = "_read"]
                unsafe fn read(fd: c_int, buf: *mut c_void, count: c_uint)
                            -> c_int;

                #[link_name = "_rmdir"]
                unsafe fn rmdir(path: *c_char) -> c_int;

                #[link_name = "_unlink"]
                unsafe fn unlink(c: *c_char) -> c_int;

                #[link_name = "_write"]
                unsafe fn write(fd: c_int, buf: *c_void, count: c_uint)
                             -> c_int;
            }
        }
    }


    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    pub mod posix88 {
        pub mod stat_ {
            use libc::types::os::arch::c95::{c_char, c_int};
            use libc::types::os::arch::posix01::stat;
            use libc::types::os::arch::posix88::mode_t;

            #[nolink]
            #[abi = "cdecl"]
            pub extern {
                unsafe fn chmod(path: *c_char, mode: mode_t) -> c_int;
                unsafe fn fchmod(fd: c_int, mode: mode_t) -> c_int;

                #[cfg(target_os = "linux")]
                #[cfg(target_os = "freebsd")]
                #[cfg(target_os = "android")]
               unsafe fn fstat(fildes: c_int, buf: *mut stat) -> c_int;

                #[cfg(target_os = "macos")]
                #[link_name = "fstat64"]
                unsafe fn fstat(fildes: c_int, buf: *mut stat) -> c_int;

                unsafe fn mkdir(path: *c_char, mode: mode_t) -> c_int;
                unsafe fn mkfifo(path: *c_char, mode: mode_t) -> c_int;

                #[cfg(target_os = "linux")]
                #[cfg(target_os = "freebsd")]
                #[cfg(target_os = "android")]
                unsafe fn stat(path: *c_char, buf: *mut stat) -> c_int;

                #[cfg(target_os = "macos")]
                #[link_name = "stat64"]
                unsafe fn stat(path: *c_char, buf: *mut stat) -> c_int;
            }
        }

        #[nolink]
        #[abi = "cdecl"]
        pub mod stdio {
            use libc::types::common::c95::FILE;
            use libc::types::os::arch::c95::{c_char, c_int};

            pub extern {
                unsafe fn popen(command: *c_char, mode: *c_char) -> *FILE;
                unsafe fn pclose(stream: *FILE) -> c_int;
                unsafe fn fdopen(fd: c_int, mode: *c_char) -> *FILE;
                unsafe fn fileno(stream: *FILE) -> c_int;
            }
        }

        #[nolink]
        #[abi = "cdecl"]
        pub mod fcntl {
            use libc::types::os::arch::c95::{c_char, c_int};
            use libc::types::os::arch::posix88::mode_t;

            pub extern {
                unsafe fn open(path: *c_char, oflag: c_int, mode: c_int)
                            -> c_int;
                unsafe fn creat(path: *c_char, mode: mode_t) -> c_int;
                unsafe fn fcntl(fd: c_int, cmd: c_int) -> c_int;
            }
        }

        #[nolink]
        #[abi = "cdecl"]
        pub mod dirent {
            use libc::types::common::posix88::{DIR, dirent_t};
            use libc::types::os::arch::c95::{c_char, c_int, c_long};

            // NOTE: On OS X opendir and readdir have two versions,
            // one for 32-bit kernelspace and one for 64.
            // We should be linking to the 64-bit ones, called
            // opendir$INODE64, etc. but for some reason rustc
            // doesn't link it correctly on i686, so we're going
            // through a C function that mysteriously does work.
            pub unsafe fn opendir(dirname: *c_char) -> *DIR {
                rust_opendir(dirname)
            }
            pub unsafe fn readdir(dirp: *DIR) -> *dirent_t {
                rust_readdir(dirp)
            }

            extern {
                unsafe fn rust_opendir(dirname: *c_char) -> *DIR;
                unsafe fn rust_readdir(dirp: *DIR) -> *dirent_t;
            }

            pub extern {
                unsafe fn closedir(dirp: *DIR) -> c_int;
                unsafe fn rewinddir(dirp: *DIR);
                unsafe fn seekdir(dirp: *DIR, loc: c_long);
                unsafe fn telldir(dirp: *DIR) -> c_long;
            }
        }

        #[nolink]
        #[abi = "cdecl"]
        pub mod unistd {
            use libc::types::common::c95::c_void;
            use libc::types::os::arch::c95::{c_char, c_int, c_long, c_uint};
            use libc::types::os::arch::c95::{size_t};
            use libc::types::os::arch::posix88::{gid_t, off_t, pid_t};
            use libc::types::os::arch::posix88::{ssize_t, uid_t};

            pub extern {
                unsafe fn access(path: *c_char, amode: c_int) -> c_int;
                unsafe fn alarm(seconds: c_uint) -> c_uint;
                unsafe fn chdir(dir: *c_char) -> c_int;
                unsafe fn chown(path: *c_char, uid: uid_t, gid: gid_t)
                             -> c_int;
                unsafe fn close(fd: c_int) -> c_int;
                unsafe fn dup(fd: c_int) -> c_int;
                unsafe fn dup2(src: c_int, dst: c_int) -> c_int;
                unsafe fn execv(prog: *c_char, argv: **c_char) -> c_int;
                unsafe fn execve(prog: *c_char,
                                 argv: **c_char,
                                 envp: **c_char)
                              -> c_int;
                unsafe fn execvp(c: *c_char, argv: **c_char) -> c_int;
                unsafe fn fork() -> pid_t;
                unsafe fn fpathconf(filedes: c_int, name: c_int) -> c_long;
                unsafe fn getcwd(buf: *c_char, size: size_t) -> *c_char;
                unsafe fn getegid() -> gid_t;
                unsafe fn geteuid() -> uid_t;
                unsafe fn getgid() -> gid_t ;
                unsafe fn getgroups(ngroups_max: c_int, groups: *mut gid_t)
                                 -> c_int;
                unsafe fn getlogin() -> *c_char;
                unsafe fn getopt(argc: c_int, argv: **c_char, optstr: *c_char)
                              -> c_int;
                unsafe fn getpgrp() -> pid_t;
                unsafe fn getpid() -> pid_t;
                unsafe fn getppid() -> pid_t;
                unsafe fn getuid() -> uid_t;
                unsafe fn isatty(fd: c_int) -> c_int;
                unsafe fn link(src: *c_char, dst: *c_char) -> c_int;
                unsafe fn lseek(fd: c_int, offset: off_t, whence: c_int)
                             -> off_t;
                unsafe fn pathconf(path: *c_char, name: c_int) -> c_long;
                unsafe fn pause() -> c_int;
                unsafe fn pipe(fds: *mut c_int) -> c_int;
                unsafe fn read(fd: c_int, buf: *mut c_void,
                        count: size_t) -> ssize_t;
                unsafe fn rmdir(path: *c_char) -> c_int;
                unsafe fn setgid(gid: gid_t) -> c_int;
                unsafe fn setpgid(pid: pid_t, pgid: pid_t) -> c_int;
                unsafe fn setsid() -> pid_t;
                unsafe fn setuid(uid: uid_t) -> c_int;
                unsafe fn sleep(secs: c_uint) -> c_uint;
                unsafe fn sysconf(name: c_int) -> c_long;
                unsafe fn tcgetpgrp(fd: c_int) -> pid_t;
                unsafe fn ttyname(fd: c_int) -> *c_char;
                unsafe fn unlink(c: *c_char) -> c_int;
                unsafe fn write(fd: c_int, buf: *c_void, count: size_t)
                             -> ssize_t;
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    pub mod posix01 {
        #[nolink]
        #[abi = "cdecl"]
        pub mod stat_ {
            use libc::types::os::arch::c95::{c_char, c_int};
            use libc::types::os::arch::posix01::stat;

            pub extern {
                #[cfg(target_os = "linux")]
                #[cfg(target_os = "freebsd")]
                #[cfg(target_os = "android")]
                unsafe fn lstat(path: *c_char, buf: *mut stat) -> c_int;

                #[cfg(target_os = "macos")]
                #[link_name = "lstat64"]
                unsafe fn lstat(path: *c_char, buf: *mut stat) -> c_int;
            }
        }

        #[nolink]
        #[abi = "cdecl"]
        pub mod unistd {
            use libc::types::os::arch::c95::{c_char, c_int, size_t};
            use libc::types::os::arch::posix88::{ssize_t};

            pub extern {
                unsafe fn readlink(path: *c_char, buf: *mut c_char,
                            bufsz: size_t) -> ssize_t;

                unsafe fn fsync(fd: c_int) -> c_int;

                #[cfg(target_os = "linux")]
                #[cfg(target_os = "android")]
                unsafe fn fdatasync(fd: c_int) -> c_int;

                unsafe fn setenv(name: *c_char, val: *c_char,
                          overwrite: c_int) -> c_int;
                unsafe fn unsetenv(name: *c_char) -> c_int;
                unsafe fn putenv(string: *c_char) -> c_int;

                unsafe fn symlink(path1: *c_char, path2: *c_char) -> c_int;
            }
        }

        #[nolink]
        #[abi = "cdecl"]
        pub mod wait {
            use libc::types::os::arch::c95::{c_int};
            use libc::types::os::arch::posix88::{pid_t};

            pub extern {
                unsafe fn waitpid(pid: pid_t,
                                  status: *mut c_int,
                                  options: c_int)
                               -> pid_t;
            }
        }
    }

    #[cfg(target_os = "win32")]
    pub mod posix01 {
        pub mod stat_ {
        }

        pub mod unistd {
        }
    }


    #[cfg(target_os = "win32")]
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    pub mod posix08 {
        pub mod unistd {
        }
    }


    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    pub mod bsd44 {
        use libc::types::common::c95::{c_void};
        use libc::types::os::arch::c95::{c_char, c_int, c_uint, size_t};

        #[abi = "cdecl"]
        pub extern {
            unsafe fn sysctl(name: *c_int, namelen: c_uint,
                      oldp: *mut c_void, oldlenp: *mut size_t,
                      newp: *c_void, newlen: size_t) -> c_int;

            unsafe fn sysctlbyname(name: *c_char,
                            oldp: *mut c_void, oldlenp: *mut size_t,
                            newp: *c_void, newlen: size_t) -> c_int;

            unsafe fn sysctlnametomib(name: *c_char, mibp: *mut c_int,
                               sizep: *mut size_t) -> c_int;
        }
    }


    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    #[cfg(target_os = "win32")]
    pub mod bsd44 {
    }

    #[cfg(target_os = "macos")]
    #[nolink]
    pub mod extra {
        use libc::types::os::arch::c95::{c_char, c_int};

        #[abi = "cdecl"]
        pub extern {
            unsafe fn _NSGetExecutablePath(buf: *mut c_char,
                                           bufsize: *mut u32)
                                        -> c_int;
        }
    }

    #[cfg(target_os = "freebsd")]
    pub mod extra {
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    pub mod extra {
    }


    #[cfg(target_os = "win32")]
    pub mod extra {

        pub mod kernel32 {
            use libc::types::os::arch::extra::{BOOL, DWORD, HMODULE};
            use libc::types::os::arch::extra::{LPCWSTR, LPWSTR, LPTCH};
            use libc::types::os::arch::extra::{LPSECURITY_ATTRIBUTES};
            use libc::types::os::arch::extra::{HANDLE};

            #[abi = "stdcall"]
            pub extern "stdcall" {
                unsafe fn GetEnvironmentVariableW(n: LPCWSTR,
                                                  v: LPWSTR,
                                                  nsize: DWORD)
                                               -> DWORD;
                unsafe fn SetEnvironmentVariableW(n: LPCWSTR, v: LPCWSTR)
                                               -> BOOL;
                unsafe fn GetEnvironmentStringsA() -> LPTCH;
                unsafe fn FreeEnvironmentStringsA(env_ptr: LPTCH) -> BOOL;

                unsafe fn GetModuleFileNameW(hModule: HMODULE,
                                             lpFilename: LPWSTR,
                                             nSize: DWORD)
                                          -> DWORD;
                unsafe fn CreateDirectoryW(lpPathName: LPCWSTR,
                                           lpSecurityAttributes:
                                           LPSECURITY_ATTRIBUTES)
                                        -> BOOL;
                unsafe fn CopyFileW(lpExistingFileName: LPCWSTR,
                                    lpNewFileName: LPCWSTR,
                                    bFailIfExists: BOOL)
                                 -> BOOL;
                unsafe fn DeleteFileW(lpPathName: LPCWSTR) -> BOOL;
                unsafe fn RemoveDirectoryW(lpPathName: LPCWSTR) -> BOOL;
                unsafe fn SetCurrentDirectoryW(lpPathName: LPCWSTR) -> BOOL;

                unsafe fn GetLastError() -> DWORD;
                unsafe fn FindFirstFileW(fileName: *u16,
                                        findFileData: HANDLE)
                    -> HANDLE;
                unsafe fn FindNextFileW(findFile: HANDLE,
                                       findFileData: HANDLE)
                    -> BOOL;
                unsafe fn FindClose(findFile: HANDLE) -> BOOL;
            }
        }

        pub mod msvcrt {
            use libc::types::os::arch::c95::c_int;

            #[abi = "cdecl"]
            #[nolink]
            pub extern {
                #[link_name = "_commit"]
                unsafe fn commit(fd: c_int) -> c_int;
            }
        }
    }
}


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
