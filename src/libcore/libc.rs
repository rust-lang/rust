// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];
/*!
* Bindings for libc.
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
* Despite having several names each, these are *reasonably* coherent
* point-in-time, list-of-definition sorts of specs. You can get each under a
* variety of names but will wind up with the same definition in each case.
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

pub use types::common::c95::*;
pub use types::common::c99::*;
pub use types::common::posix88::*;
pub use types::common::posix01::*;
pub use types::common::posix08::*;
pub use types::common::bsd44::*;
pub use types::os::common::posix01::*;
pub use types::os::arch::c95::*;
pub use types::os::arch::c99::*;
pub use types::os::arch::posix88::*;
pub use types::os::arch::posix01::*;
pub use types::os::arch::posix08::*;
pub use types::os::arch::bsd44::*;
pub use types::os::arch::extra::*;

pub use consts::os::c95::*;
pub use consts::os::c99::*;
pub use consts::os::posix88::*;
pub use consts::os::posix01::*;
pub use consts::os::posix08::*;
pub use consts::os::bsd44::*;
pub use consts::os::extra::*;

pub use funcs::c95::ctype::*;
pub use funcs::c95::stdio::*;
pub use funcs::c95::stdlib::*;
pub use funcs::c95::string::*;

pub use funcs::posix88::stat_::*;
pub use funcs::posix88::stdio::*;
pub use funcs::posix88::fcntl::*;
pub use funcs::posix88::dirent::*;
pub use funcs::posix88::unistd::*;

pub use funcs::posix01::stat_::*;
pub use funcs::posix01::unistd::*;
pub use funcs::posix08::unistd::*;

pub use funcs::bsd44::*;
pub use funcs::extra::*;

// Explicit export lists for the intersection (provided here) mean that
// you can write more-platform-agnostic code if you stick to just these
// symbols.

pub use size_t;
pub use c_float, c_double, c_void, FILE, fpos_t;
pub use DIR, dirent_t;
pub use c_char, c_schar, c_uchar;
pub use c_short, c_ushort, c_int, c_uint, c_long, c_ulong;
pub use size_t, ptrdiff_t, clock_t, time_t;
pub use c_longlong, c_ulonglong, intptr_t, uintptr_t;
pub use off_t, dev_t, ino_t, pid_t, mode_t, ssize_t;

pub use EXIT_FAILURE, EXIT_SUCCESS, RAND_MAX,
EOF, SEEK_SET, SEEK_CUR, SEEK_END, _IOFBF, _IONBF, _IOLBF,
BUFSIZ, FOPEN_MAX, FILENAME_MAX, L_tmpnam, TMP_MAX,
O_RDONLY, O_WRONLY, O_RDWR, O_APPEND, O_CREAT, O_EXCL, O_TRUNC,
S_IFIFO, S_IFCHR, S_IFBLK, S_IFDIR, S_IFREG, S_IFMT, S_IEXEC,
S_IWRITE, S_IREAD, S_IRWXU, S_IXUSR, S_IWUSR, S_IRUSR, F_OK, R_OK,
W_OK, X_OK, STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO;

pub use isalnum, isalpha, iscntrl, isdigit, islower, isprint, ispunct,
isspace, isupper, isxdigit, tolower, toupper;

pub use fopen, freopen, fflush, fclose, remove, tmpfile, setvbuf, setbuf,
fgetc, fgets, fputc, fputs, puts, ungetc, fread, fwrite, fseek, ftell,
rewind, fgetpos, fsetpos, feof, ferror, perror;

pub use abs, labs, atof, atoi, strtod, strtol, strtoul, calloc, malloc,
realloc, free, abort, exit, system, getenv, rand, srand;

pub use strcpy, strncpy, strcat, strncat, strcmp, strncmp, strcoll, strchr,
strrchr, strspn, strcspn, strpbrk, strstr, strlen, strerror, strtok,
strxfrm, memcpy, memmove, memcmp, memchr, memset;

pub use chmod, mkdir;
pub use popen, pclose, fdopen, fileno;
pub use open, creat;
pub use access, chdir, close, dup, dup2, execv, execve, execvp, getcwd,
getpid, isatty, lseek, pipe, read, rmdir, unlink, write;

pub use fstat, lstat, stat;


mod types {

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
    pub mod os {
        pub mod common {
            pub mod posix01 {}
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
                    __unused: [c_long * 3],
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
                    __unused: [uint8_t * 2],
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
                pub struct stat {
                    st_dev: dev_t,
                    st_ino: ino_t,
                    st_mode: mode_t,
                    st_nlink: c_short,
                    st_uid: c_short,
                    st_gid: c_short,
                    st_rdev: dev_t,
                    st_size: int64_t,
                    st_atime: time64_t,
                    st_mtime: time64_t,
                    st_c_time: time64_t,
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
            }
        }
    }

    #[cfg(target_os = "macos")]
    pub mod os {
        pub mod common {
            pub mod posix01 {
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
                    st_qspare: [int64_t * 2],
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
            pub const EXIT_FAILURE : int = 1;
            pub const EXIT_SUCCESS : int = 0;
            pub const RAND_MAX : int = 32767;
            pub const EOF : int = -1;
            pub const SEEK_SET : int = 0;
            pub const SEEK_CUR : int = 1;
            pub const SEEK_END : int = 2;
            pub const _IOFBF : int = 0;
            pub const _IONBF : int = 4;
            pub const _IOLBF : int = 64;
            pub const BUFSIZ : uint = 512_u;
            pub const FOPEN_MAX : uint = 20_u;
            pub const FILENAME_MAX : uint = 260_u;
            pub const L_tmpnam : uint = 16_u;
            pub const TMP_MAX : uint = 32767_u;
        }
        pub mod c99 {
        }
        pub mod posix88 {
            pub const O_RDONLY : int = 0;
            pub const O_WRONLY : int = 1;
            pub const O_RDWR : int = 2;
            pub const O_APPEND : int = 8;
            pub const O_CREAT : int = 256;
            pub const O_EXCL : int = 1024;
            pub const O_TRUNC : int = 512;
            pub const S_IFIFO : int = 4096;
            pub const S_IFCHR : int = 8192;
            pub const S_IFBLK : int = 12288;
            pub const S_IFDIR : int = 16384;
            pub const S_IFREG : int = 32768;
            pub const S_IFMT : int = 61440;
            pub const S_IEXEC : int = 64;
            pub const S_IWRITE : int = 128;
            pub const S_IREAD : int = 256;
            pub const S_IRWXU : int = 448;
            pub const S_IXUSR : int = 64;
            pub const S_IWUSR : int = 128;
            pub const S_IRUSR : int = 256;
            pub const F_OK : int = 0;
            pub const R_OK : int = 4;
            pub const W_OK : int = 2;
            pub const X_OK : int = 1;
            pub const STDIN_FILENO : int = 0;
            pub const STDOUT_FILENO : int = 1;
            pub const STDERR_FILENO : int = 2;
        }
        pub mod posix01 {
        }
        pub mod posix08 {
        }
        pub mod bsd44 {
        }
        pub mod extra {
            pub const O_TEXT : int = 16384;
            pub const O_BINARY : int = 32768;
            pub const O_NOINHERIT: int = 128;

            pub const ERROR_SUCCESS : int = 0;
            pub const ERROR_INSUFFICIENT_BUFFER : int = 122;
        }
    }


    #[cfg(target_os = "linux")]
    pub mod os {
        pub mod c95 {
            pub const EXIT_FAILURE : int = 1;
            pub const EXIT_SUCCESS : int = 0;
            pub const RAND_MAX : int = 2147483647;
            pub const EOF : int = -1;
            pub const SEEK_SET : int = 0;
            pub const SEEK_CUR : int = 1;
            pub const SEEK_END : int = 2;
            pub const _IOFBF : int = 0;
            pub const _IONBF : int = 2;
            pub const _IOLBF : int = 1;
            pub const BUFSIZ : uint = 8192_u;
            pub const FOPEN_MAX : uint = 16_u;
            pub const FILENAME_MAX : uint = 4096_u;
            pub const L_tmpnam : uint = 20_u;
            pub const TMP_MAX : uint = 238328_u;
        }
        pub mod c99 {
        }
        pub mod posix88 {
            pub const O_RDONLY : int = 0;
            pub const O_WRONLY : int = 1;
            pub const O_RDWR : int = 2;
            pub const O_APPEND : int = 1024;
            pub const O_CREAT : int = 64;
            pub const O_EXCL : int = 128;
            pub const O_TRUNC : int = 512;
            pub const S_IFIFO : int = 4096;
            pub const S_IFCHR : int = 8192;
            pub const S_IFBLK : int = 24576;
            pub const S_IFDIR : int = 16384;
            pub const S_IFREG : int = 32768;
            pub const S_IFMT : int = 61440;
            pub const S_IEXEC : int = 64;
            pub const S_IWRITE : int = 128;
            pub const S_IREAD : int = 256;
            pub const S_IRWXU : int = 448;
            pub const S_IXUSR : int = 64;
            pub const S_IWUSR : int = 128;
            pub const S_IRUSR : int = 256;
            pub const F_OK : int = 0;
            pub const R_OK : int = 4;
            pub const W_OK : int = 2;
            pub const X_OK : int = 1;
            pub const STDIN_FILENO : int = 0;
            pub const STDOUT_FILENO : int = 1;
            pub const STDERR_FILENO : int = 2;
            pub const F_LOCK : int = 1;
            pub const F_TEST : int = 3;
            pub const F_TLOCK : int = 2;
            pub const F_ULOCK : int = 0;
        }
        pub mod posix01 {
        }
        pub mod posix08 {
        }
        pub mod bsd44 {
        }
        pub mod extra {
            pub const O_RSYNC : int = 1052672;
            pub const O_DSYNC : int = 4096;
            pub const O_SYNC : int = 1052672;
        }
    }

    #[cfg(target_os = "freebsd")]
    pub mod os {
        pub mod c95 {
            pub const EXIT_FAILURE : int = 1;
            pub const EXIT_SUCCESS : int = 0;
            pub const RAND_MAX : int = 2147483647;
            pub const EOF : int = -1;
            pub const SEEK_SET : int = 0;
            pub const SEEK_CUR : int = 1;
            pub const SEEK_END : int = 2;
            pub const _IOFBF : int = 0;
            pub const _IONBF : int = 2;
            pub const _IOLBF : int = 1;
            pub const BUFSIZ : uint = 1024_u;
            pub const FOPEN_MAX : uint = 20_u;
            pub const FILENAME_MAX : uint = 1024_u;
            pub const L_tmpnam : uint = 1024_u;
            pub const TMP_MAX : uint = 308915776_u;
        }
        pub mod c99 {
        }
        pub mod posix88 {
            pub const O_RDONLY : int = 0;
            pub const O_WRONLY : int = 1;
            pub const O_RDWR : int = 2;
            pub const O_APPEND : int = 8;
            pub const O_CREAT : int = 512;
            pub const O_EXCL : int = 2048;
            pub const O_TRUNC : int = 1024;
            pub const S_IFIFO : int = 4096;
            pub const S_IFCHR : int = 8192;
            pub const S_IFBLK : int = 24576;
            pub const S_IFDIR : int = 16384;
            pub const S_IFREG : int = 32768;
            pub const S_IFMT : int = 61440;
            pub const S_IEXEC : int = 64;
            pub const S_IWRITE : int = 128;
            pub const S_IREAD : int = 256;
            pub const S_IRWXU : int = 448;
            pub const S_IXUSR : int = 64;
            pub const S_IWUSR : int = 128;
            pub const S_IRUSR : int = 256;
            pub const F_OK : int = 0;
            pub const R_OK : int = 4;
            pub const W_OK : int = 2;
            pub const X_OK : int = 1;
            pub const STDIN_FILENO : int = 0;
            pub const STDOUT_FILENO : int = 1;
            pub const STDERR_FILENO : int = 2;
            pub const F_LOCK : int = 1;
            pub const F_TEST : int = 3;
            pub const F_TLOCK : int = 2;
            pub const F_ULOCK : int = 0;
        }
        pub mod posix01 {
        }
        pub mod posix08 {
        }
        pub mod bsd44 {
        }
        pub mod extra {
            pub const O_SYNC : int = 128;
            pub const CTL_KERN: int = 1;
            pub const KERN_PROC: int = 14;
            pub const KERN_PROC_PATHNAME: int = 12;
        }
    }

    #[cfg(target_os = "macos")]
    pub mod os {
        pub mod c95 {
            pub const EXIT_FAILURE : int = 1;
            pub const EXIT_SUCCESS : int = 0;
            pub const RAND_MAX : int = 2147483647;
            pub const EOF : int = -1;
            pub const SEEK_SET : int = 0;
            pub const SEEK_CUR : int = 1;
            pub const SEEK_END : int = 2;
            pub const _IOFBF : int = 0;
            pub const _IONBF : int = 2;
            pub const _IOLBF : int = 1;
            pub const BUFSIZ : uint = 1024_u;
            pub const FOPEN_MAX : uint = 20_u;
            pub const FILENAME_MAX : uint = 1024_u;
            pub const L_tmpnam : uint = 1024_u;
            pub const TMP_MAX : uint = 308915776_u;
        }
        pub mod c99 {
        }
        pub mod posix88 {
            pub const O_RDONLY : int = 0;
            pub const O_WRONLY : int = 1;
            pub const O_RDWR : int = 2;
            pub const O_APPEND : int = 8;
            pub const O_CREAT : int = 512;
            pub const O_EXCL : int = 2048;
            pub const O_TRUNC : int = 1024;
            pub const S_IFIFO : int = 4096;
            pub const S_IFCHR : int = 8192;
            pub const S_IFBLK : int = 24576;
            pub const S_IFDIR : int = 16384;
            pub const S_IFREG : int = 32768;
            pub const S_IFMT : int = 61440;
            pub const S_IEXEC : int = 64;
            pub const S_IWRITE : int = 128;
            pub const S_IREAD : int = 256;
            pub const S_IRWXU : int = 448;
            pub const S_IXUSR : int = 64;
            pub const S_IWUSR : int = 128;
            pub const S_IRUSR : int = 256;
            pub const F_OK : int = 0;
            pub const R_OK : int = 4;
            pub const W_OK : int = 2;
            pub const X_OK : int = 1;
            pub const STDIN_FILENO : int = 0;
            pub const STDOUT_FILENO : int = 1;
            pub const STDERR_FILENO : int = 2;
            pub const F_LOCK : int = 1;
            pub const F_TEST : int = 3;
            pub const F_TLOCK : int = 2;
            pub const F_ULOCK : int = 0;
        }
        pub mod posix01 {
        }
        pub mod posix08 {
        }
        pub mod bsd44 {
        }
        pub mod extra {
            pub const O_DSYNC : int = 4194304;
            pub const O_SYNC : int = 128;
            pub const F_FULLFSYNC : int = 51;
        }
    }
}


pub mod funcs {
    // Thankfull most of c95 is universally available and does not vary by OS
    // or anything. The same is not true of POSIX.

    pub mod c95 {
        #[nolink]
        #[abi = "cdecl"]
        pub extern mod ctype {
            fn isalnum(c: c_int) -> c_int;
            fn isalpha(c: c_int) -> c_int;
            fn iscntrl(c: c_int) -> c_int;
            fn isdigit(c: c_int) -> c_int;
            fn isgraph(c: c_int) -> c_int;
            fn islower(c: c_int) -> c_int;
            fn isprint(c: c_int) -> c_int;
            fn ispunct(c: c_int) -> c_int;
            fn isspace(c: c_int) -> c_int;
            fn isupper(c: c_int) -> c_int;
            fn isxdigit(c: c_int) -> c_int;
            fn tolower(c: c_char) -> c_char;
            fn toupper(c: c_char) -> c_char;
        }

        #[nolink]
        #[abi = "cdecl"]
        pub extern mod stdio {
            fn fopen(filename: *c_char, mode: *c_char) -> *FILE;
            fn freopen(filename: *c_char, mode: *c_char,
                       file: *FILE) -> *FILE;
            fn fflush(file: *FILE) -> c_int;
            fn fclose(file: *FILE) -> c_int;
            fn remove(filename: *c_char) -> c_int;
            fn rename(oldname: *c_char, newname: *c_char) -> c_int;
            fn tmpfile() -> *FILE;
            fn setvbuf(stream: *FILE, buffer: *c_char,
                       mode: c_int, size: size_t) -> c_int;
            fn setbuf(stream: *FILE, buf: *c_char);
            // Omitted: printf and scanf variants.
            fn fgetc(stream: *FILE) -> c_int;
            fn fgets(buf: *mut c_char, n: c_int,
                     stream: *FILE) -> *c_char;
            fn fputc(c: c_int, stream: *FILE) -> c_int;
            fn fputs(s: *c_char, stream: *FILE) -> *c_char;
            // Omitted: getc, getchar (might be macros).

            // Omitted: gets, so ridiculously unsafe that it should not
            // survive.

            // Omitted: putc, putchar (might be macros).
            fn puts(s: *c_char) -> c_int;
            fn ungetc(c: c_int, stream: *FILE) -> c_int;
            fn fread(ptr: *mut c_void, size: size_t,
                     nobj: size_t, stream: *FILE) -> size_t;
            fn fwrite(ptr: *c_void, size: size_t,
                      nobj: size_t, stream: *FILE) -> size_t;
            fn fseek(stream: *FILE, offset: c_long,
                     whence: c_int) -> c_int;
            fn ftell(stream: *FILE) -> c_long;
            fn rewind(stream: *FILE);
            fn fgetpos(stream: *FILE, ptr: *fpos_t) -> c_int;
            fn fsetpos(stream: *FILE, ptr: *fpos_t) -> c_int;
            fn feof(stream: *FILE) -> c_int;
            fn ferror(stream: *FILE) -> c_int;
            fn perror(s: *c_char);
        }


        #[nolink]
        #[abi = "cdecl"]
        pub extern mod stdlib {
            fn abs(i: c_int) -> c_int;
            fn labs(i: c_long) -> c_long;
            // Omitted: div, ldiv (return pub type incomplete).
            fn atof(s: *c_char) -> c_double;
            fn atoi(s: *c_char) -> c_int;
            fn strtod(s: *c_char, endp: **c_char) -> c_double;
            fn strtol(s: *c_char, endp: **c_char, base: c_int) -> c_long;
            fn strtoul(s: *c_char, endp: **c_char,
                       base: c_int) -> c_ulong;
            fn calloc(nobj: size_t, size: size_t) -> *c_void;
            fn malloc(size: size_t) -> *c_void;
            fn realloc(p: *c_void, size: size_t) -> *c_void;
            fn free(p: *c_void);
            fn abort() -> !;
            fn exit(status: c_int) -> !;
            // Omitted: atexit.
            fn system(s: *c_char) -> c_int;
            fn getenv(s: *c_char) -> *c_char;
            // Omitted: bsearch, qsort
            fn rand() -> c_int;
            fn srand(seed: c_uint);
        }

        #[nolink]
        #[abi = "cdecl"]
        pub extern mod string {
            fn strcpy(dst: *c_char, src: *c_char) -> *c_char;
            fn strncpy(dst: *c_char, src: *c_char, n: size_t) -> *c_char;
            fn strcat(s: *c_char, ct: *c_char) -> *c_char;
            fn strncat(s: *c_char, ct: *c_char, n: size_t) -> *c_char;
            fn strcmp(cs: *c_char, ct: *c_char) -> c_int;
            fn strncmp(cs: *c_char, ct: *c_char, n: size_t) -> c_int;
            fn strcoll(cs: *c_char, ct: *c_char) -> c_int;
            fn strchr(cs: *c_char, c: c_int) -> *c_char;
            fn strrchr(cs: *c_char, c: c_int) -> *c_char;
            fn strspn(cs: *c_char, ct: *c_char) -> size_t;
            fn strcspn(cs: *c_char, ct: *c_char) -> size_t;
            fn strpbrk(cs: *c_char, ct: *c_char) -> *c_char;
            fn strstr(cs: *c_char, ct: *c_char) -> *c_char;
            fn strlen(cs: *c_char) -> size_t;
            fn strerror(n: c_int) -> *c_char;
            fn strtok(s: *c_char, t: *c_char) -> *c_char;
            fn strxfrm(s: *c_char, ct: *c_char, n: size_t) -> size_t;
            fn memcpy(s: *c_void, ct: *c_void, n: size_t) -> *c_void;
            fn memmove(s: *c_void, ct: *c_void, n: size_t) -> *c_void;
            fn memcmp(cx: *c_void, ct: *c_void, n: size_t) -> c_int;
            fn memchr(cx: *c_void, c: c_int, n: size_t) -> *c_void;
            fn memset(s: *c_void, c: c_int, n: size_t) -> *c_void;
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
        pub extern mod stat_ {
            #[link_name = "_chmod"]
            fn chmod(path: *c_char, mode: c_int) -> c_int;

            #[link_name = "_mkdir"]
            fn mkdir(path: *c_char) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        pub extern mod stdio {
            #[link_name = "_popen"]
            fn popen(command: *c_char, mode: *c_char) -> *FILE;

            #[link_name = "_pclose"]
            fn pclose(stream: *FILE) -> c_int;

            #[link_name = "_fdopen"]
            fn fdopen(fd: c_int, mode: *c_char) -> *FILE;

            #[link_name = "_fileno"]
            fn fileno(stream: *FILE) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        pub extern mod fcntl {
            #[link_name = "_open"]
            fn open(path: *c_char, oflag: c_int, mode: c_int) -> c_int;

            #[link_name = "_creat"]
            fn creat(path: *c_char, mode: c_int) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        pub extern mod dirent {
            // Not supplied at all.
        }

        #[nolink]
        #[abi = "cdecl"]
        pub extern mod unistd {
            #[link_name = "_access"]
            fn access(path: *c_char, amode: c_int) -> c_int;

            #[link_name = "_chdir"]
            fn chdir(dir: *c_char) -> c_int;

            #[link_name = "_close"]
            fn close(fd: c_int) -> c_int;

            #[link_name = "_dup"]
            fn dup(fd: c_int) -> c_int;

            #[link_name = "_dup2"]
            fn dup2(src: c_int, dst: c_int) -> c_int;

            #[link_name = "_execv"]
            fn execv(prog: *c_char, argv: **c_char) -> intptr_t;

            #[link_name = "_execve"]
            fn execve(prog: *c_char, argv: **c_char,
                      envp: **c_char) -> c_int;

            #[link_name = "_execvp"]
            fn execvp(c: *c_char, argv: **c_char) -> c_int;

            #[link_name = "_execvpe"]
            fn execvpe(c: *c_char, argv: **c_char,
                       envp: **c_char) -> c_int;

            #[link_name = "_getcwd"]
            fn getcwd(buf: *c_char, size: size_t) -> *c_char;

            #[link_name = "_getpid"]
            fn getpid() -> c_int;

            #[link_name = "_isatty"]
            fn isatty(fd: c_int) -> c_int;

            #[link_name = "_lseek"]
            fn lseek(fd: c_int, offset: c_long, origin: c_int) -> c_long;

            #[link_name = "_pipe"]
            fn pipe(fds: *mut c_int, psize: c_uint,
                    textmode: c_int) -> c_int;

            #[link_name = "_read"]
            fn read(fd: c_int, buf: *mut c_void, count: c_uint) -> c_int;

            #[link_name = "_rmdir"]
            fn rmdir(path: *c_char) -> c_int;

            #[link_name = "_unlink"]
            fn unlink(c: *c_char) -> c_int;

            #[link_name = "_write"]
            fn write(fd: c_int, buf: *c_void, count: c_uint) -> c_int;

        }
    }


    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    pub mod posix88 {
        #[nolink]
        #[abi = "cdecl"]
        pub extern mod stat_ {
            fn chmod(path: *c_char, mode: mode_t) -> c_int;
            fn fchmod(fd: c_int, mode: mode_t) -> c_int;

            #[cfg(target_os = "linux")]
            #[cfg(target_os = "freebsd")]
            fn fstat(fildes: c_int, buf: *mut stat) -> c_int;

            #[cfg(target_os = "macos")]
            #[link_name = "fstat64"]
            fn fstat(fildes: c_int, buf: *mut stat) -> c_int;

            fn mkdir(path: *c_char, mode: mode_t) -> c_int;
            fn mkfifo(path: *c_char, mode: mode_t) -> c_int;

            #[cfg(target_os = "linux")]
            #[cfg(target_os = "freebsd")]
            fn stat(path: *c_char, buf: *mut stat) -> c_int;

            #[cfg(target_os = "macos")]
            #[link_name = "stat64"]
            fn stat(path: *c_char, buf: *mut stat) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        pub extern mod stdio {
            fn popen(command: *c_char, mode: *c_char) -> *FILE;
            fn pclose(stream: *FILE) -> c_int;
            fn fdopen(fd: c_int, mode: *c_char) -> *FILE;
            fn fileno(stream: *FILE) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        pub extern mod fcntl {
            fn open(path: *c_char, oflag: c_int, mode: c_int) -> c_int;
            fn creat(path: *c_char, mode: mode_t) -> c_int;
            fn fcntl(fd: c_int, cmd: c_int) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        pub extern mod dirent {
            fn opendir(dirname: *c_char) -> *DIR;
            fn closedir(dirp: *DIR) -> c_int;
            fn readdir(dirp: *DIR) -> *dirent_t;
            fn rewinddir(dirp: *DIR);
            fn seekdir(dirp: *DIR, loc: c_long);
            fn telldir(dirp: *DIR) -> c_long;
        }

        #[nolink]
        #[abi = "cdecl"]
        pub extern mod unistd {
            fn access(path: *c_char, amode: c_int) -> c_int;
            fn alarm(seconds: c_uint) -> c_uint;
            fn chdir(dir: *c_char) -> c_int;
            fn chown(path: *c_char, uid: uid_t, gid: gid_t) -> c_int;
            fn close(fd: c_int) -> c_int;
            fn dup(fd: c_int) -> c_int;
            fn dup2(src: c_int, dst: c_int) -> c_int;
            fn execv(prog: *c_char, argv: **c_char) -> c_int;
            fn execve(prog: *c_char, argv: **c_char,
                      envp: **c_char) -> c_int;
            fn execvp(c: *c_char, argv: **c_char) -> c_int;
            fn fork() -> pid_t;
            fn fpathconf(filedes: c_int, name: c_int) -> c_long;
            fn getcwd(buf: *c_char, size: size_t) -> *c_char;
            fn getegid() -> gid_t;
            fn geteuid() -> uid_t;
            fn getgid() -> gid_t ;
            fn getgroups(ngroups_max: c_int, groups: *mut gid_t) -> c_int;
            fn getlogin() -> *c_char;
            fn getopt(argc: c_int, argv: **c_char,
                      optstr: *c_char) -> c_int;
            fn getpgrp() -> pid_t;
            fn getpid() -> pid_t;
            fn getppid() -> pid_t;
            fn getuid() -> uid_t;
            fn isatty(fd: c_int) -> c_int;
            fn link(src: *c_char, dst: *c_char) -> c_int;
            fn lseek(fd: c_int, offset: off_t, whence: c_int) -> off_t;
            fn pathconf(path: *c_char, name: c_int) -> c_long;
            fn pause() -> c_int;
            fn pipe(fds: *mut c_int) -> c_int;
            fn read(fd: c_int, buf: *mut c_void,
                    count: size_t) -> ssize_t;
            fn rmdir(path: *c_char) -> c_int;
            fn setgid(gid: gid_t) -> c_int;
            fn setpgid(pid: pid_t, pgid: pid_t) -> c_int;
            fn setsid() -> pid_t;
            fn setuid(uid: uid_t) -> c_int;
            fn sleep(secs: c_uint) -> c_uint;
            fn sysconf(name: c_int) -> c_long;
            fn tcgetpgrp(fd: c_int) -> pid_t;
            fn ttyname(fd: c_int) -> *c_char;
            fn unlink(c: *c_char) -> c_int;
            fn write(fd: c_int, buf: *c_void, count: size_t) -> ssize_t;
        }
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    pub mod posix01 {
        #[nolink]
        #[abi = "cdecl"]
        pub extern mod stat_ {
            #[cfg(target_os = "linux")]
            #[cfg(target_os = "freebsd")]
            fn lstat(path: *c_char, buf: *mut stat) -> c_int;

            #[cfg(target_os = "macos")]
            #[link_name = "lstat64"]
            fn lstat(path: *c_char, buf: *mut stat) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        pub extern mod unistd {
            fn readlink(path: *c_char, buf: *mut c_char,
                        bufsz: size_t) -> ssize_t;

            fn fsync(fd: c_int) -> c_int;

            #[cfg(target_os = "linux")]
            fn fdatasync(fd: c_int) -> c_int;

            fn setenv(name: *c_char, val: *c_char,
                      overwrite: c_int) -> c_int;
            fn unsetenv(name: *c_char) -> c_int;
            fn putenv(string: *c_char) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        pub extern mod wait {
            fn waitpid(pid: pid_t, status: *mut c_int,
                       options: c_int) -> pid_t;
        }
    }

    #[cfg(target_os = "win32")]
    pub mod posix01 {
        #[nolink]
        pub extern mod stat_ {
        }

        #[nolink]
        pub extern mod unistd {
        }
    }


    #[cfg(target_os = "win32")]
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    pub mod posix08 {
        #[nolink]
        pub extern mod unistd {
        }
    }


    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    #[nolink]
    #[abi = "cdecl"]
    pub extern mod bsd44 {
        fn sysctl(name: *c_int, namelen: c_uint,
                  oldp: *mut c_void, oldlenp: *mut size_t,
                  newp: *c_void, newlen: size_t) -> c_int;

        fn sysctlbyname(name: *c_char,
                        oldp: *mut c_void, oldlenp: *mut size_t,
                        newp: *c_void, newlen: size_t) -> c_int;

        fn sysctlnametomib(name: *c_char, mibp: *mut c_int,
                           sizep: *mut size_t) -> c_int;
    }


    #[cfg(target_os = "linux")]
    #[cfg(target_os = "win32")]
    pub mod bsd44 {
    }


    #[cfg(target_os = "macos")]
    #[nolink]
    #[abi = "cdecl"]
    pub extern mod extra {
        fn _NSGetExecutablePath(buf: *mut c_char,
                                bufsize: *mut u32) -> c_int;
    }

    #[cfg(target_os = "freebsd")]
    pub mod extra {
    }

    #[cfg(target_os = "linux")]
    pub mod extra {
    }


    #[cfg(target_os = "win32")]
    pub mod extra {
        use types::os::arch::extra::*;
        pub use kernel32::*;
        pub use msvcrt::*;

        #[abi = "stdcall"]
        pub extern mod kernel32 {
            fn GetEnvironmentVariableW(n: LPCWSTR,
                                       v: LPWSTR,
                                       nsize: DWORD) -> DWORD;
            fn SetEnvironmentVariableW(n: LPCWSTR, v: LPCWSTR) -> BOOL;

            fn GetModuleFileNameW(hModule: HMODULE,
                                  lpFilename: LPWSTR,
                                  nSize: DWORD) -> DWORD;
            fn CreateDirectoryW(lpPathName: LPCWSTR,
                                lpSecurityAttributes:
                                LPSECURITY_ATTRIBUTES) -> BOOL;
            fn CopyFileW(lpExistingFileName: LPCWSTR,
                         lpNewFileName: LPCWSTR,
                         bFailIfExists: BOOL) -> BOOL;
            fn DeleteFileW(lpPathName: LPCWSTR) -> BOOL;
            fn RemoveDirectoryW(lpPathName: LPCWSTR) -> BOOL;
            fn SetCurrentDirectoryW(lpPathName: LPCWSTR) -> BOOL;

            fn GetLastError() -> DWORD;
        }

        #[abi = "cdecl"]
        #[nolink]
        pub extern mod msvcrt {
            #[link_name = "_commit"]
            fn commit(fd: c_int) -> c_int;
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
