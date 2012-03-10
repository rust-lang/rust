//
// We consider the following specs reasonably normative with respect
// to interoperating with the C standard library (libc/msvcrt):
//
//   - ISO 9899:1990 ('C95', 'ANSI C', 'Standard C'), NA1, 1995.
//   - ISO 9899:1999 ('C99' or 'C9x').
//   - ISO 9945:1988 / IEEE 1003.1-1988 ('POSIX.1').
//   - ISO 9945:2001 / IEEE 1003.1-2001 ('POSIX:2001', 'SUSv3').
//   - ISO 9945:2008 / IEEE 1003.1-2008 ('POSIX:2008', 'SUSv4').
//
// Despite having several names each, these are *reasonably* coherent
// point-in-time, list-of-definition sorts of specs. You can get each under a
// variety of names but will wind up with the same definition in each case.
//
// Our interface to these libraries is complicated by the non-universality of
// conformance to any of them. About the only thing universally supported is
// the first (C95), beyond that definitions quickly become absent on various
// platforms.
//
// We therefore wind up dividing our module-space up (mostly for the sake of
// sanity while editing, filling-in-details and eliminating duplication) into
// definitions common-to-all (held in modules named c95, c99, posix88, posix01
// and posix08) and definitions that appear only on *some* platforms (named
// 'extra'). This would be things like significant OSX foundation kit, or
// win32 library kernel32.dll, or various fancy glibc, linux or BSD
// extensions.
//
// In addition to the per-platform 'extra' modules, we define a module of
// "common BSD" libc routines that never quite made it into POSIX but show up
// in multiple derived systems. This is the 4.4BSD r2 / 1995 release, the
// final one from Berkeley after the lawsuits died down and the CSRG
// dissolved.
//

// Initial glob-exports mean that all the contents of all the modules
// wind up exported, if you're interested in writing platform-specific code.

// FIXME: change these to glob-exports when sufficiently supported.

import types::common::c95::*;
import types::common::c99::*;
import types::common::posix88::*;
import types::common::posix01::*;
import types::common::posix08::*;
import types::common::bsd44::*;
import types::os::arch::c95::*;
import types::os::arch::c99::*;
import types::os::arch::posix88::*;
import types::os::arch::posix01::*;
import types::os::arch::posix08::*;
import types::os::arch::bsd44::*;
import types::os::arch::extra::*;

import consts::os::c95::*;
import consts::os::c99::*;
import consts::os::posix88::*;
import consts::os::posix01::*;
import consts::os::posix08::*;
import consts::os::bsd44::*;
import consts::os::extra::*;

import funcs::c95::ctype::*;
import funcs::c95::stdio::*;
import funcs::c95::stdlib::*;
import funcs::c95::string::*;

import funcs::posix88::stat::*;
import funcs::posix88::stdio::*;
import funcs::posix88::fcntl::*;
import funcs::posix88::dirent::*;
import funcs::posix88::unistd::*;

import funcs::posix01::unistd::*;
import funcs::posix08::unistd::*;

import funcs::bsd44::*;
import funcs::extra::*;

// FIXME: remove these 3 exports (and their uses next door in os::) when
// export globs work. They provide access (for now) for os:: to dig around in
// the rest of the platform-specific definitions.

export types, funcs, consts;

// Explicit export lists for the intersection (provided here) mean that
// you can write more-platform-agnostic code if you stick to just these
// symbols.

export c_float, c_double, c_void, FILE, fpos_t;
export DIR, dirent;
export c_char, c_schar, c_uchar;
export c_short, c_ushort, c_int, c_uint, c_long, c_ulong;
export size_t, ptrdiff_t, clock_t, time_t;
export c_longlong, c_ulonglong, intptr_t, uintptr_t;
export off_t, dev_t, ino_t, pid_t, mode_t, ssize_t;

export isalnum, isalpha, iscntrl, isdigit, islower, isprint, ispunct,
       isspace, isupper, isxdigit, tolower, toupper;

export fopen, freopen, fflush, fclose, remove, tmpfile, setvbuf, setbuf,
       fgetc, fgets, fputc, fputs, puts, ungetc, fread, fwrite, fseek, ftell,
       rewind, fgetpos, fsetpos, feof, ferror, perror;

export abs, labs, atof, atoi, strtod, strtol, strtoul, calloc, malloc,
       realloc, free, abort, exit, system, getenv, rand, srand;

export strcpy, strncpy, strcat, strncat, strcmp, strncmp, strcoll, strchr,
       strrchr, strspn, strcspn, strpbrk, strstr, strlen, strerror, strtok,
       strxfrm, memcpy, memmove, memcmp, memchr, memset;

export chmod, mkdir;
export popen, pclose, fdopen;
export open, creat;
export access, chdir, close, dup, dup2, execv, execve, execvp, getcwd,
       getpid, isatty, lseek, pipe, read, rmdir, unlink, write;


mod types {

    // Types tend to vary *per architecture* so we pull their definitions out
    // into this module.

    // Standard types that are opaque or common, so are not per-target.
    mod common {
        mod c95 {
            enum c_void {}
            enum FILE {}
            enum fpos_t {}
        }
        mod c99 { }
        mod posix88 {
            enum DIR {}
            enum dirent {}
        }
        mod posix01 { }
        mod posix08 { }
        mod bsd44 { }
    }

    // Standard types that are scalar but vary by OS and arch.

    #[cfg(target_os = "linux")]
    mod os {
        #[cfg(target_arch = "x86")]
        mod arch {
            mod c95 {
                type c_char = i8;
                type c_schar = i8;
                type c_uchar = u8;
                type c_short = i16;
                type c_ushort = u16;
                type c_int = i32;
                type c_uint = u32;
                type c_long = i32;
                type c_ulong = u32;
                type c_float = f32;
                type c_double = f64;
                type size_t = u32;
                type ptrdiff_t = i32;
                type clock_t = i32;
                type time_t = i32;
                type wchar_t = i32;
            }
            mod c99 {
                type c_longlong = i64;
                type c_ulonglong = u64;
                type intptr_t = i32;
                type uintptr_t = u32;
            }
            mod posix88 {
                type off_t = i32;
                type dev_t = u64;
                type ino_t = u32;
                type pid_t = i32;
                type uid_t = u32;
                type gid_t = u32;
                type useconds_t = u32;
                type mode_t = u32;
                type ssize_t = i32;
            }
            mod posix01 { }
            mod posix08 { }
            mod bsd44 { }
            mod extra {
            }
        }

        #[cfg(target_arch = "x86_64")]
        mod arch {
            mod c95 {
                type c_char = i8;
                type c_schar = i8;
                type c_uchar = u8;
                type c_short = i16;
                type c_ushort = u16;
                type c_int = i32;
                type c_uint = u32;
                type c_long = i64;
                type c_ulong = u64;
                type c_float = f32;
                type c_double = f64;
                type size_t = u64;
                type ptrdiff_t = i64;
                type clock_t = i64;
                type time_t = i64;
                type wchar_t = i32;
            }
            mod c99 {
                type c_longlong = i64;
                type c_ulonglong = u64;
                type intptr_t = i64;
                type uintptr_t = u64;
            }
            mod posix88 {
                type off_t = i64;
                type dev_t = u64;
                type ino_t = u64;
                type pid_t = i32;
                type uid_t = u32;
                type gid_t = u32;
                type useconds_t = u32;
                type mode_t = u32;
                type ssize_t = i64;
            }
            mod posix01 { }
            mod posix08 { }
            mod bsd44 { }
            mod extra {
            }
        }
    }

    #[cfg(target_os = "freebsd")]
    mod os {
        #[cfg(target_arch = "x86_64")]
        mod arch {
            mod c95 {
                type c_char = i8;
                type c_schar = i8;
                type c_uchar = u8;
                type c_short = i16;
                type c_ushort = u16;
                type c_int = i32;
                type c_uint = u32;
                type c_long = i64;
                type c_ulong = u64;
                type c_float = f32;
                type c_double = f64;
                type size_t = u64;
                type ptrdiff_t = i64;
                type clock_t = i32;
                type time_t = i64;
                type wchar_t = i32;
            }
            mod c99 {
                type c_longlong = i64;
                type c_ulonglong = u64;
                type intptr_t = i64;
                type uintptr_t = u64;
            }
            mod posix88 {
                type off_t = i64;
                type dev_t = u32;
                type ino_t = u32;
                type pid_t = i32;
                type uid_t = u32;
                type gid_t = u32;
                type useconds_t = u32;
                type mode_t = u16;
                type ssize_t = i64;
            }
            mod posix01 { }
            mod posix08 { }
            mod bsd44 { }
            mod extra {
            }
        }
    }

    #[cfg(target_os = "win32")]
    mod os {
        #[cfg(target_arch = "x86")]
        mod arch {
            mod c95 {
                type c_char = i8;
                type c_schar = i8;
                type c_uchar = u8;
                type c_short = i16;
                type c_ushort = u16;
                type c_int = i32;
                type c_uint = u32;
                type c_long = i32;
                type c_ulong = u32;
                type c_float = f32;
                type c_double = f64;
                type size_t = u32;
                type ptrdiff_t = i32;
                type clock_t = i32;
                type time_t = i32;
                type wchar_t = u16;
            }
            mod c99 {
                type c_longlong = i64;
                type c_ulonglong = u64;
                type intptr_t = i32;
                type uintptr_t = u32;
            }
            mod posix88 {
                type off_t = i32;
                type dev_t = u32;
                type ino_t = i16;
                type pid_t = i32;
                type useconds_t = u32;
                type mode_t = u16;
                type ssize_t = i32;
            }
            mod posix01 { }
            mod posix08 { }
            mod bsd44 { }
            mod extra {
                type BOOL = c_int;
                type BYTE = u8;
                type CCHAR = c_char;
                type CHAR = c_char;

                type DWORD = c_ulong;
                type DWORDLONG = c_ulonglong;

                type HANDLE = LPVOID;
                type HMODULE = c_uint;

                type LONG_PTR = c_long;

                type LPCWSTR = *WCHAR;
                type LPCSTR = *CHAR;

                type LPWSTR = *mutable WCHAR;
                type LPSTR = *mutable CHAR;

                // Not really, but opaque to us.
                type LPSECURITY_ATTRIBUTES = LPVOID;

                type LPVOID = *mutable c_void;
                type LPWORD = *mutable WORD;

                type LRESULT = LONG_PTR;
                type PBOOL = *mutable BOOL;
                type WCHAR = wchar_t;
                type WORD = u16;
            }
        }
    }

    #[cfg(target_os = "macos")]
    mod os {
        #[cfg(target_arch = "x86")]
        mod arch {
            mod c95 {
                type c_char = i8;
                type c_schar = i8;
                type c_uchar = u8;
                type c_short = i16;
                type c_ushort = u16;
                type c_int = i32;
                type c_uint = u32;
                type c_long = i32;
                type c_ulong = u32;
                type c_float = f32;
                type c_double = f64;
                type size_t = u32;
                type ptrdiff_t = i32;
                type clock_t = u32;
                type time_t = i32;
                type wchar_t = i32;
            }
            mod c99 {
                type c_longlong = i64;
                type c_ulonglong = u64;
                type intptr_t = i32;
                type uintptr_t = u32;
            }
            mod posix88 {
                type off_t = i64;
                type dev_t = i32;
                type ino_t = u64;
                type pid_t = i32;
                type uid_t = u32;
                type gid_t = u32;
                type useconds_t = u32;
                type mode_t = u16;
                type ssize_t = i32;
            }
            mod posix01 { }
            mod posix08 { }
            mod bsd44 { }
            mod extra {
            }
        }

        #[cfg(target_arch = "x86_64")]
        mod arch {
            mod c95 {
                type c_char = i8;
                type c_schar = i8;
                type c_uchar = u8;
                type c_short = i16;
                type c_ushort = u16;
                type c_int = i32;
                type c_uint = u32;
                type c_long = i64;
                type c_ulong = u64;
                type c_float = f32;
                type c_double = f64;
                type size_t = u64;
                type ptrdiff_t = i64;
                type clock_t = u64;
                type time_t = i64;
                type wchar_t = i32;
            }
            mod c99 {
                type c_longlong = i64;
                type c_ulonglong = u64;
                type intptr_t = i64;
                type uintptr_t = u64;
            }
            mod posix88 {
                type off_t = i64;
                type dev_t = i32;
                type ino_t = u64;
                type pid_t = i32;
                type uid_t = u32;
                type gid_t = u32;
                type useconds_t = u32;
                type mode_t = u16;
                type ssize_t = i64;
            }
            mod posix01 { }
            mod posix08 { }
            mod bsd44 { }
            mod extra {
            }
        }
    }
}

mod consts {

    // Consts tend to vary per OS so we pull their definitions out
    // into this module.

    #[cfg(target_os = "win32")]
    mod os {
        mod c95 {
            const EXIT_FAILURE : int = 1;
            const EXIT_SUCCESS : int = 0;
            const RAND_MAX : int = 32767;
            const EOF : int = -1;
            const SEEK_SET : int = 0;
            const SEEK_CUR : int = 1;
            const SEEK_END : int = 2;
            const _IOFBF : int = 0;
            const _IONBF : int = 4;
            const _IOLBF : int = 64;
            const BUFSIZ : uint = 512_u;
            const FOPEN_MAX : uint = 20_u;
            const FILENAME_MAX : uint = 260_u;
            const L_tmpnam : uint = 16_u;
            const TMP_MAX : uint = 32767_u;
        }
        mod c99 { }
        mod posix88 {
            const O_RDONLY : int = 0;
            const O_WRONLY : int = 1;
            const O_RDWR : int = 2;
            const O_APPEND : int = 8;
            const O_CREAT : int = 256;
            const O_EXCL : int = 1024;
            const O_TRUNC : int = 512;
            const S_IFIFO : int = 4096;
            const S_IFCHR : int = 8192;
            const S_IFBLK : int = 12288;
            const S_IFDIR : int = 16384;
            const S_IFREG : int = 32768;
            const S_IFMT : int = 61440;
            const S_IEXEC : int = 64;
            const S_IWRITE : int = 128;
            const S_IREAD : int = 256;
            const S_IRWXU : int = 448;
            const S_IXUSR : int = 64;
            const S_IWUSR : int = 128;
            const S_IRUSR : int = 256;
            const F_OK : int = 0;
            const R_OK : int = 4;
            const W_OK : int = 2;
            const X_OK : int = 1;
            const STDERR_FILENO : int = 2;
            const STDIN_FILENO : int = 0;
            const STDOUT_FILENO : int = 1;
        }
        mod posix01 { }
        mod posix08 { }
        mod bsd44 { }
        mod extra {
            const O_TEXT : int = 16384;
            const O_BINARY : int = 32768;
            const O_NOINHERIT: int = 128;

            const ERROR_SUCCESS : int = 0;
            const ERROR_INSUFFICIENT_BUFFER : int = 122;
        }
    }


    #[cfg(target_os = "linux")]
    mod os {
        mod c95 {
            const EXIT_FAILURE : int = 1;
            const EXIT_SUCCESS : int = 0;
            const RAND_MAX : int = 2147483647;
            const EOF : int = -1;
            const SEEK_SET : int = 0;
            const SEEK_CUR : int = 1;
            const SEEK_END : int = 2;
            const _IOFBF : int = 0;
            const _IONBF : int = 2;
            const _IOLBF : int = 1;
            const BUFSIZ : uint = 8192_u;
            const FOPEN_MAX : uint = 16_u;
            const FILENAME_MAX : uint = 4096_u;
            const L_tmpnam : uint = 20_u;
            const TMP_MAX : uint = 238328_u;
        }
        mod c99 { }
        mod posix88 {
            const O_RDONLY : int = 0;
            const O_WRONLY : int = 1;
            const O_RDWR : int = 2;
            const O_APPEND : int = 1024;
            const O_CREAT : int = 64;
            const O_EXCL : int = 128;
            const O_TRUNC : int = 512;
            const S_IFIFO : int = 4096;
            const S_IFCHR : int = 8192;
            const S_IFBLK : int = 24576;
            const S_IFDIR : int = 16384;
            const S_IFREG : int = 32768;
            const S_IFMT : int = 61440;
            const S_IEXEC : int = 64;
            const S_IWRITE : int = 128;
            const S_IREAD : int = 256;
            const S_IRWXU : int = 448;
            const S_IXUSR : int = 64;
            const S_IWUSR : int = 128;
            const S_IRUSR : int = 256;
            const F_OK : int = 0;
            const R_OK : int = 4;
            const W_OK : int = 2;
            const X_OK : int = 1;
            const F_LOCK : int = 1;
            const F_TEST : int = 3;
            const F_TLOCK : int = 2;
            const F_ULOCK : int = 0;
        }
        mod posix01 { }
        mod posix08 { }
        mod bsd44 { }
        mod extra {
            const O_RSYNC : int = 1052672;
            const O_DSYNC : int = 4096;
            const O_SYNC : int = 1052672;
        }
    }

    #[cfg(target_os = "freebsd")]
    mod os {
        mod c95 {
            const EXIT_FAILURE : int = 1;
            const EXIT_SUCCESS : int = 0;
            const RAND_MAX : int = 2147483647;
            const EOF : int = -1;
            const SEEK_SET : int = 0;
            const SEEK_CUR : int = 1;
            const SEEK_END : int = 2;
            const _IOFBF : int = 0;
            const _IONBF : int = 2;
            const _IOLBF : int = 1;
            const BUFSIZ : uint = 1024_u;
            const FOPEN_MAX : uint = 20_u;
            const FILENAME_MAX : uint = 1024_u;
            const L_tmpnam : uint = 1024_u;
            const TMP_MAX : uint = 308915776_u;
        }
        mod c99 { }
        mod posix88 {
            const O_RDONLY : int = 0;
            const O_WRONLY : int = 1;
            const O_RDWR : int = 2;
            const O_APPEND : int = 8;
            const O_CREAT : int = 512;
            const O_EXCL : int = 2048;
            const O_TRUNC : int = 1024;
            const S_IFIFO : int = 4096;
            const S_IFCHR : int = 8192;
            const S_IFBLK : int = 24576;
            const S_IFDIR : int = 16384;
            const S_IFREG : int = 32768;
            const S_IFMT : int = 61440;
            const S_IEXEC : int = 64;
            const S_IWRITE : int = 128;
            const S_IREAD : int = 256;
            const S_IRWXU : int = 448;
            const S_IXUSR : int = 64;
            const S_IWUSR : int = 128;
            const S_IRUSR : int = 256;
            const F_OK : int = 0;
            const R_OK : int = 4;
            const W_OK : int = 2;
            const X_OK : int = 1;
            const STDERR_FILENO : int = 2;
            const STDIN_FILENO : int = 0;
            const STDOUT_FILENO : int = 1;
            const F_LOCK : int = 1;
            const F_TEST : int = 3;
            const F_TLOCK : int = 2;
            const F_ULOCK : int = 0;
        }
        mod posix01 { }
        mod posix08 { }
        mod bsd44 { }
        mod extra {
            const O_SYNC : int = 128;
            const CTL_KERN: int = 1;
            const KERN_PROC: int = 14;
            const KERN_PROC_PATHNAME: int = 12;
        }
    }

    #[cfg(target_os = "macos")]
    mod os {
        mod c95 {
            const EXIT_FAILURE : int = 1;
            const EXIT_SUCCESS : int = 0;
            const RAND_MAX : int = 2147483647;
            const EOF : int = -1;
            const SEEK_SET : int = 0;
            const SEEK_CUR : int = 1;
            const SEEK_END : int = 2;
            const _IOFBF : int = 0;
            const _IONBF : int = 2;
            const _IOLBF : int = 1;
            const BUFSIZ : uint = 1024_u;
            const FOPEN_MAX : uint = 20_u;
            const FILENAME_MAX : uint = 1024_u;
            const L_tmpnam : uint = 1024_u;
            const TMP_MAX : uint = 308915776_u;
        }
        mod c99 { }
        mod posix88 {
            const O_RDONLY : int = 0;
            const O_WRONLY : int = 1;
            const O_RDWR : int = 2;
            const O_APPEND : int = 8;
            const O_CREAT : int = 512;
            const O_EXCL : int = 2048;
            const O_TRUNC : int = 1024;
            const S_IFIFO : int = 4096;
            const S_IFCHR : int = 8192;
            const S_IFBLK : int = 24576;
            const S_IFDIR : int = 16384;
            const S_IFREG : int = 32768;
            const S_IFMT : int = 61440;
            const S_IEXEC : int = 64;
            const S_IWRITE : int = 128;
            const S_IREAD : int = 256;
            const S_IRWXU : int = 448;
            const S_IXUSR : int = 64;
            const S_IWUSR : int = 128;
            const S_IRUSR : int = 256;
            const F_OK : int = 0;
            const R_OK : int = 4;
            const W_OK : int = 2;
            const X_OK : int = 1;
            const STDERR_FILENO : int = 2;
            const STDIN_FILENO : int = 0;
            const STDOUT_FILENO : int = 1;
            const F_LOCK : int = 1;
            const F_TEST : int = 3;
            const F_TLOCK : int = 2;
            const F_ULOCK : int = 0;
        }
        mod posix01 { }
        mod posix08 { }
        mod bsd44 { }
        mod extra {
            const O_DSYNC : int = 4194304;
            const O_SYNC : int = 128;
            const F_FULLFSYNC : int = 51;
        }
    }
}


mod funcs {

    // Thankfull most of c95 is universally available and does not vary by OS
    // or anything. The same is not true of POSIX.

    mod c95 {

        #[nolink]
        #[abi = "cdecl"]
        native mod ctype {
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
            fn tolower(c: c_int) -> c_int;
            fn toupper(c: c_int) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        native mod stdio {

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
            fn fgets(buf: *c_char, n: c_int, stream: *FILE) -> *c_char;
            fn fputc(c: c_int, stream: *FILE) -> c_int;
            fn fputs(s: *c_char, stream: *FILE) -> *c_char;
            // Omitted: getc, getchar (might be macros).

            // Omitted: gets, so ridiculously unsafe that it should not
            // survive.

            // Omitted: putc, putchar (might be macros).
            fn puts(s: *c_char) -> c_int;
            fn ungetc(c: c_int, stream: *FILE) -> c_int;
            fn fread(ptr: *c_void, size: size_t,
                     nobj: size_t, stream: *FILE) -> size_t;
            fn fwrite(ptr: *c_void, size: size_t,
                      nobj: size_t, stream: *FILE) -> size_t;
            fn fseek(stream: *FILE, offset: c_long, whence: c_int) -> c_int;
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
        native mod stdlib {
            fn abs(i: c_int) -> c_int;
            fn labs(i: c_long) -> c_long;
            // Omitted: div, ldiv (return type incomplete).
            fn atof(s: *c_char) -> c_double;
            fn atoi(s: *c_char) -> c_int;
            fn strtod(s: *c_char, endp: **c_char) -> c_double;
            fn strtol(s: *c_char, endp: **c_char, base: c_int) -> c_long;
            fn strtoul(s: *c_char, endp: **c_char, base: c_int) -> c_ulong;
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
        native mod string {

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
    mod posix88 {

        #[nolink]
        #[abi = "cdecl"]
        native mod stat {
            #[link_name = "_chmod"]
            fn chmod(path: *c_char, mode: c_int) -> c_int;

            #[link_name = "_mkdir"]
            fn mkdir(path: *c_char) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        native mod stdio {
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
        native mod fcntl {
            #[link_name = "_open"]
            fn open(path: *c_char, oflag: c_int) -> c_int;

            #[link_name = "_creat"]
            fn creat(path: *c_char, mode: c_int) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        native mod dirent {
            // Not supplied at all.
        }

        #[nolink]
        #[abi = "cdecl"]
        native mod unistd {
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
            fn execvpe(c: *c_char, argv: **c_char, envp: **c_char) -> c_int;

            #[link_name = "_getcwd"]
            fn getcwd(buf: *c_char, size: size_t) -> *c_char;

            #[link_name = "_getpid"]
            fn getpid() -> c_int;

            #[link_name = "_isatty"]
            fn isatty(fd: c_int) -> c_int;

            #[link_name = "_lseek"]
            fn lseek(fd: c_int, offset: c_long, origin: c_int) -> c_long;

            #[link_name = "_pipe"]
            fn pipe(fds: *mutable c_int, psize: c_uint,
                    textmode: c_int) -> c_int;

            #[link_name = "_read"]
            fn read(fd: c_int, buf: *c_void, count: c_uint) -> c_int;

            #[link_name = "_rmdir"]
            fn rmdir(path: *c_char) -> c_int;

            #[link_name = "_unlink"]
            fn unlink(c: *c_char) -> c_int;

            #[link_name = "_write"]
            fn write(fd: c_int, buf: *c_void, count: c_uint) -> c_uint;

        }
    }


    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    mod posix88 {

        #[nolink]
        #[abi = "cdecl"]
        native mod stat {
            fn chmod(path: *c_char, mode: mode_t) -> c_int;
            fn fchmod(fd: c_int, mode: mode_t) -> c_int;
            fn mkdir(path: *c_char, mode: mode_t) -> c_int;
            fn mkfifo(ath: *c_char, mode: mode_t) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        native mod stdio {
            fn popen(command: *c_char, mode: *c_char) -> *FILE;
            fn pclose(stream: *FILE) -> c_int;
            fn fdopen(fd: c_int, mode: *c_char) -> *FILE;
            fn fileno(stream: *FILE) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        native mod fcntl {
            fn open(path: *c_char, oflag: c_int) -> c_int;
            fn creat(path: *c_char, mode: mode_t) -> c_int;
            fn fcntl(fd: c_int, cmd: c_int) -> c_int;
        }

        #[nolink]
        #[abi = "cdecl"]
        native mod dirent {
            fn opendir(dirname: *c_char) -> *DIR;
            fn closedir(dirp: *DIR) -> c_int;
            fn readdir(dirp: *DIR) -> *dirent;
            fn rewinddir(dirp: *DIR);
            fn seekdir(dirp: *DIR, loc: c_long);
            fn telldir(dirp: *DIR) -> c_long;
        }

        #[nolink]
        #[abi = "cdecl"]
        native mod unistd {
            fn access(path: *c_char, amode: c_int) -> c_int;
            fn alarm(seconds: c_uint) -> c_uint;
            fn chdir(dir: *c_char) -> c_int;
            fn chown(path: *c_char, uid: uid_t, gid: gid_t) -> c_int;
            fn close(fd: c_int) -> c_int;
            fn dup(fd: c_int) -> c_int;
            fn dup2(src: c_int, dst: c_int) -> c_int;
            fn execv(prog: *c_char, argv: **c_char) -> c_int;
            fn execve(prog: *c_char, argv: **c_char, envp: **c_char) -> c_int;
            fn execvp(c: *c_char, argv: **c_char) -> c_int;
            fn fork() -> pid_t;
            fn fpathconf(filedes: c_int, name: c_int) -> c_long;
            fn getcwd(buf: *c_char, size: size_t) -> *c_char;
            fn getegid() -> gid_t;
            fn geteuid() -> uid_t;
            fn getgid() -> gid_t ;
            fn getgroups(ngroups_max: c_int, groups: *gid_t) -> c_int;
            fn getlogin() -> *c_char;
            fn getopt(argc: c_int, argv: **c_char, optstr: *c_char) -> c_int;
            fn getpgrp() -> pid_t;
            fn getpid() -> pid_t;
            fn getppid() -> pid_t;
            fn getuid() -> uid_t;
            fn isatty(fd: c_int) -> c_int;
            fn link(src: *c_char, dst: *c_char) -> c_int;
            fn lseek(fd: c_int, offset: off_t, whence: c_int) -> off_t;
            fn pathconf(path: *c_char, name: c_int) -> c_long;
            fn pause() -> c_int;
            fn pipe(fds: *mutable c_int) -> c_int;
            fn read(fd: c_int, buf: *c_void, count: size_t) -> ssize_t;
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
    mod posix01 {

        #[nolink]
        #[abi = "cdecl"]
        native mod unistd {
            fn readlink(path: *c_char, buf: *mutable c_char,
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
        native mod wait {
            fn waitpid(pid: pid_t, status: *mutable c_int,
                       options: c_int) -> pid_t;
        }
    }

    #[cfg(target_os = "win32")]
    mod posix01 {
        #[nolink]
        native mod unistd { }
    }


    #[cfg(target_os = "win32")]
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    mod posix08 {
        #[nolink]
        native mod unistd { }
    }


    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    #[nolink]
    #[abi = "cdecl"]
    native mod bsd44 {

        fn sysctl(name: *c_int, namelen: c_uint,
                  oldp: *mutable c_void, oldlenp: *mutable size_t,
                  newp: *c_void, newlen: size_t) -> c_int;

        fn sysctlbyname(name: *c_char,
                        oldp: *mutable c_void, oldlenp: *mutable size_t,
                        newp: *c_void, newlen: size_t) -> c_int;

        fn sysctlnametomib(name: *c_char, mibp: *mutable c_int,
                           sizep: *mutable size_t) -> c_int;
    }


    #[cfg(target_os = "linux")]
    #[cfg(target_os = "win32")]
    mod bsd44 {
    }


    #[cfg(target_os = "macos")]
    #[nolink]
    #[abi = "cdecl"]
    native mod extra {
        fn _NSGetExecutablePath(buf: *mutable c_char,
                                bufsize: *mutable u32) -> c_int;
    }

    #[cfg(target_os = "freebsd")]
    mod extra { }

    #[cfg(target_os = "linux")]
    mod extra { }


    #[cfg(target_os = "win32")]
    mod extra {
        import types::os::arch::extra::*;

        #[abi = "stdcall"]
        native mod kernel32 {
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
            fn DeleteFileW(lpPathName: LPCWSTR) -> BOOL;
            fn RemoveDirectoryW(lpPathName: LPCWSTR) -> BOOL;
            fn SetCurrentDirectoryW(lpPathName: LPCWSTR) -> BOOL;

            fn GetLastError() -> DWORD;
        }

        #[abi = "cdecl"]
        #[nolink]
        native mod msvcrt {
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
