// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Crate docs

#![allow(bad_style, overflowing_literals, improper_ctypes)]
#![crate_type = "rlib"]
#![crate_name = "libc"]
#![cfg_attr(dox, feature(no_core, lang_items))]
#![cfg_attr(dox, no_core)]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico")]

#![cfg_attr(all(target_os = "linux", target_arch = "x86_64"), doc(
    html_root_url = "https://doc.rust-lang.org/libc/x86_64-unknown-linux-gnu"
))]
#![cfg_attr(all(target_os = "linux", target_arch = "x86"), doc(
    html_root_url = "https://doc.rust-lang.org/libc/i686-unknown-linux-gnu"
))]
#![cfg_attr(all(target_os = "linux", target_arch = "arm"), doc(
    html_root_url = "https://doc.rust-lang.org/libc/arm-unknown-linux-gnueabihf"
))]
#![cfg_attr(all(target_os = "linux", target_arch = "mips"), doc(
    html_root_url = "https://doc.rust-lang.org/libc/mips-unknown-linux-gnu"
))]
#![cfg_attr(all(target_os = "linux", target_arch = "aarch64"), doc(
    html_root_url = "https://doc.rust-lang.org/libc/aarch64-unknown-linux-gnu"
))]
#![cfg_attr(all(target_os = "linux", target_env = "musl"), doc(
    html_root_url = "https://doc.rust-lang.org/libc/x86_64-unknown-linux-musl"
))]
#![cfg_attr(all(target_os = "macos", target_arch = "x86_64"), doc(
    html_root_url = "https://doc.rust-lang.org/libc/x86_64-apple-darwin"
))]
#![cfg_attr(all(target_os = "macos", target_arch = "x86"), doc(
    html_root_url = "https://doc.rust-lang.org/libc/i686-apple-darwin"
))]
#![cfg_attr(all(windows, target_arch = "x86_64", target_env = "gnu"), doc(
    html_root_url = "https://doc.rust-lang.org/libc/x86_64-pc-windows-gnu"
))]
#![cfg_attr(all(windows, target_arch = "x86", target_env = "gnu"), doc(
    html_root_url = "https://doc.rust-lang.org/libc/i686-pc-windows-gnu"
))]
#![cfg_attr(all(windows, target_arch = "x86_64", target_env = "msvc"), doc(
    html_root_url = "https://doc.rust-lang.org/libc/x86_64-pc-windows-msvc"
))]
#![cfg_attr(all(windows, target_arch = "x86", target_env = "msvc"), doc(
    html_root_url = "https://doc.rust-lang.org/libc/i686-pc-windows-msvc"
))]
#![cfg_attr(target_os = "android", doc(
    html_root_url = "https://doc.rust-lang.org/libc/arm-linux-androideabi"
))]
#![cfg_attr(target_os = "freebsd", doc(
    html_root_url = "https://doc.rust-lang.org/libc/x86_64-unknown-freebsd"
))]
#![cfg_attr(target_os = "openbsd", doc(
    html_root_url = "https://doc.rust-lang.org/libc/x86_64-unknown-openbsd"
))]
#![cfg_attr(target_os = "bitrig", doc(
    html_root_url = "https://doc.rust-lang.org/libc/x86_64-unknown-bitrig"
))]
#![cfg_attr(target_os = "netbsd", doc(
    html_root_url = "https://doc.rust-lang.org/libc/x86_64-unknown-netbsd"
))]
#![cfg_attr(target_os = "dragonfly", doc(
    html_root_url = "https://doc.rust-lang.org/libc/x86_64-unknown-dragonfly"
))]

// Attributes needed when building as part of the standard library
#![cfg_attr(stdbuild, feature(no_std, core, core_slice_ext, staged_api, custom_attribute, cfg_target_vendor))]
#![cfg_attr(stdbuild, no_std)]
#![cfg_attr(stdbuild, staged_api)]
#![cfg_attr(stdbuild, allow(warnings))]
#![cfg_attr(stdbuild, unstable(feature = "libc",
                               reason = "use `libc` from crates.io",
                               issue = "27783"))]

#![cfg_attr(not(feature = "use_std"), no_std)]

#[cfg(all(not(stdbuild), not(dox), feature = "use_std"))]
extern crate std as core;

#[macro_use] mod macros;
mod dox;

// Use repr(u8) as LLVM expects `void*` to be the same as `i8*` to help enable
// more optimization opportunities around it recognizing things like
// malloc/free.
#[repr(u8)]
pub enum c_void {
    // Two dummy variants so the #[repr] attribute can be used.
    #[doc(hidden)]
    __variant1,
    #[doc(hidden)]
    __variant2,
}

pub type int8_t = i8;
pub type int16_t = i16;
pub type int32_t = i32;
pub type int64_t = i64;
pub type uint8_t = u8;
pub type uint16_t = u16;
pub type uint32_t = u32;
pub type uint64_t = u64;

pub type c_schar = i8;
pub type c_uchar = u8;
pub type c_short = i16;
pub type c_ushort = u16;
pub type c_int = i32;
pub type c_uint = u32;
pub type c_float = f32;
pub type c_double = f64;
pub type c_longlong = i64;
pub type c_ulonglong = u64;
pub type intmax_t = i64;
pub type uintmax_t = u64;

pub type size_t = usize;
pub type ptrdiff_t = isize;
pub type intptr_t = isize;
pub type uintptr_t = usize;
pub type ssize_t = isize;

pub enum FILE {}
pub enum fpos_t {} // TODO: fill this out with a struct

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
    pub fn tolower(c: c_int) -> c_int;
    pub fn toupper(c: c_int) -> c_int;

    #[cfg_attr(all(target_os = "macos", target_arch = "x86"),
               link_name = "fopen$UNIX2003")]
    pub fn fopen(filename: *const c_char,
                 mode: *const c_char) -> *mut FILE;
    #[cfg_attr(all(target_os = "macos", target_arch = "x86"),
               link_name = "freopen$UNIX2003")]
    pub fn freopen(filename: *const c_char, mode: *const c_char,
                   file: *mut FILE) -> *mut FILE;
    pub fn fflush(file: *mut FILE) -> c_int;
    pub fn fclose(file: *mut FILE) -> c_int;
    pub fn remove(filename: *const c_char) -> c_int;
    pub fn rename(oldname: *const c_char, newname: *const c_char) -> c_int;
    pub fn tmpfile() -> *mut FILE;
    pub fn setvbuf(stream: *mut FILE,
                   buffer: *mut c_char,
                   mode: c_int,
                   size: size_t) -> c_int;
    pub fn setbuf(stream: *mut FILE, buf: *mut c_char);
    pub fn getchar() -> c_int;
    pub fn putchar(c: c_int) -> c_int;
    pub fn fgetc(stream: *mut FILE) -> c_int;
    pub fn fgets(buf: *mut c_char, n: c_int, stream: *mut FILE) -> *mut c_char;
    pub fn fputc(c: c_int, stream: *mut FILE) -> c_int;
    #[cfg_attr(all(target_os = "macos", target_arch = "x86"),
               link_name = "fputs$UNIX2003")]
    pub fn fputs(s: *const c_char, stream: *mut FILE)-> c_int;
    pub fn puts(s: *const c_char) -> c_int;
    pub fn ungetc(c: c_int, stream: *mut FILE) -> c_int;
    pub fn fread(ptr: *mut c_void,
                 size: size_t,
                 nobj: size_t,
                 stream: *mut FILE)
                 -> size_t;
    #[cfg_attr(all(target_os = "macos", target_arch = "x86"),
               link_name = "fwrite$UNIX2003")]
    pub fn fwrite(ptr: *const c_void,
                  size: size_t,
                  nobj: size_t,
                  stream: *mut FILE)
                  -> size_t;
    pub fn fseek(stream: *mut FILE, offset: c_long, whence: c_int) -> c_int;
    pub fn ftell(stream: *mut FILE) -> c_long;
    pub fn rewind(stream: *mut FILE);
    #[cfg_attr(target_os = "netbsd", link_name = "__fgetpos50")]
    pub fn fgetpos(stream: *mut FILE, ptr: *mut fpos_t) -> c_int;
    #[cfg_attr(target_os = "netbsd", link_name = "__fsetpos50")]
    pub fn fsetpos(stream: *mut FILE, ptr: *const fpos_t) -> c_int;
    pub fn feof(stream: *mut FILE) -> c_int;
    pub fn ferror(stream: *mut FILE) -> c_int;
    pub fn perror(s: *const c_char);
    pub fn atoi(s: *const c_char) -> c_int;
    #[cfg_attr(all(target_os = "macos", target_arch = "x86"),
               link_name = "strtod$UNIX2003")]
    pub fn strtod(s: *const c_char, endp: *mut *mut c_char) -> c_double;
    pub fn strtol(s: *const c_char,
                  endp: *mut *mut c_char, base: c_int) -> c_long;
    pub fn strtoul(s: *const c_char, endp: *mut *mut c_char,
                   base: c_int) -> c_ulong;
    pub fn calloc(nobj: size_t, size: size_t) -> *mut c_void;
    pub fn malloc(size: size_t) -> *mut c_void;
    pub fn realloc(p: *mut c_void, size: size_t) -> *mut c_void;
    pub fn free(p: *mut c_void);
    pub fn abort() -> !;
    pub fn exit(status: c_int) -> !;
    pub fn _exit(status: c_int) -> !;
    pub fn atexit(cb: extern fn()) -> c_int;
    #[cfg_attr(all(target_os = "macos", target_arch = "x86"),
               link_name = "system$UNIX2003")]
    pub fn system(s: *const c_char) -> c_int;
    pub fn getenv(s: *const c_char) -> *mut c_char;

    pub fn strcpy(dst: *mut c_char, src: *const c_char) -> *mut c_char;
    pub fn strncpy(dst: *mut c_char, src: *const c_char, n: size_t)
                   -> *mut c_char;
    pub fn strcat(s: *mut c_char, ct: *const c_char) -> *mut c_char;
    pub fn strncat(s: *mut c_char, ct: *const c_char, n: size_t) -> *mut c_char;
    pub fn strcmp(cs: *const c_char, ct: *const c_char) -> c_int;
    pub fn strncmp(cs: *const c_char, ct: *const c_char, n: size_t) -> c_int;
    pub fn strcoll(cs: *const c_char, ct: *const c_char) -> c_int;
    pub fn strchr(cs: *const c_char, c: c_int) -> *mut c_char;
    pub fn strrchr(cs: *const c_char, c: c_int) -> *mut c_char;
    pub fn strspn(cs: *const c_char, ct: *const c_char) -> size_t;
    pub fn strcspn(cs: *const c_char, ct: *const c_char) -> size_t;
    pub fn strdup(cs: *const c_char) -> *mut c_char;
    pub fn strpbrk(cs: *const c_char, ct: *const c_char) -> *mut c_char;
    pub fn strstr(cs: *const c_char, ct: *const c_char) -> *mut c_char;
    pub fn strlen(cs: *const c_char) -> size_t;
    pub fn strnlen(cs: *const c_char, maxlen: size_t) -> size_t;
    #[cfg_attr(all(target_os = "macos", target_arch = "x86"),
               link_name = "strerror$UNIX2003")]
    pub fn strerror(n: c_int) -> *mut c_char;
    pub fn strtok(s: *mut c_char, t: *const c_char) -> *mut c_char;
    pub fn strxfrm(s: *mut c_char, ct: *const c_char, n: size_t) -> size_t;
    pub fn wcslen(buf: *const wchar_t) -> size_t;

    pub fn memchr(cx: *const c_void, c: c_int, n: size_t) -> *mut c_void;
    pub fn memcmp(cx: *const c_void, ct: *const c_void, n: size_t) -> c_int;
    pub fn memcpy(dest: *mut c_void, src: *const c_void, n: size_t) -> *mut c_void;
    pub fn memmove(dest: *mut c_void, src: *const c_void, n: size_t) -> *mut c_void;
    pub fn memset(dest: *mut c_void, c: c_int, n: size_t) -> *mut c_void;
}

// These are all inline functions on android, so they end up just being entirely
// missing on that platform.
#[cfg(not(target_os = "android"))]
extern {
    pub fn abs(i: c_int) -> c_int;
    pub fn atof(s: *const c_char) -> c_double;
    pub fn labs(i: c_long) -> c_long;
    pub fn rand() -> c_int;
    pub fn srand(seed: c_uint);
}

cfg_if! {
    if #[cfg(windows)] {
        mod windows;
        pub use windows::*;
    } else if #[cfg(unix)] {
        mod unix;
        pub use unix::*;
    } else {
        // Unknown target_family
    }
}
