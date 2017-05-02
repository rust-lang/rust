// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Windows-specific primitives

#![stable(feature = "raw_ext", since = "1.1.0")]

use os::raw::{c_void, c_char, c_schar, c_uchar, c_short, c_ushort,
              c_int, c_uint, c_long, c_ulong, c_longlong, c_ulonglong};


#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PVOID = *mut c_void;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type LPVOID = *mut c_void;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type LPCVOID = *const c_void;

#[stable(feature = "raw_ext", since = "1.1.0")] pub type HANDLE = PVOID;

#[cfg(target_pointer_width = "32")]
#[stable(feature = "raw_ext", since = "1.1.0")] pub type SOCKET = u32;
#[cfg(target_pointer_width = "64")]
#[stable(feature = "raw_ext", since = "1.1.0")] pub type SOCKET = u64;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type INT = c_int;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PINT = *mut INT;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type INT8 = c_schar;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PINT8 = *mut INT8;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type INT16 = c_short;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PINT16 = *mut INT16;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type INT32 = c_int;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PINT32 = *mut INT32;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type INT64 = c_longlong;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PINT64 = *mut INT64;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type UINT = c_uint;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PUINT = *mut UINT;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type UINT8 = c_uchar;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PUINT8 = *mut UINT8;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type UINT16 = c_ushort;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PUINT16 = *mut IUNT16;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type UINT32 = c_uint;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PUINT32 = *mut UINT32;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type UINT64 = c_ulonglong;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PUINT64 = *mut UINT64;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type BOOL = c_int;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PBOOL = *mut BOOL;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type LPBOOL = *mut BOOL;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type BOOLEAN = c_int;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PBOOLEAN = *mut BOOLEAN;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type WORD = c_ushort;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PWORD = *mut WORD;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type LPWORD = *mut WORD;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type DWORD = c_ulong;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PDWORD = *mut DWORD;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type LPDWORD = *mut DWORD;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type DWORD32 = c_uint;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PDWORD32 = *mut DWORD32;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type SHORT = c_short;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PSHORT = *mut SHORT;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type USHORT = c_ushort;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PUSHORT = *mut USHORT;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type ATOM = WORD;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type UCHAR = c_uchar;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PUCHAR = *mut UCHAR;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type CHAR = c_char;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type CCHAR = CHAR;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PCHAR = *mut CHAR;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type COLORREF = DWORD;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type LPCOLORREF = *mut COLORREF;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type DWORDLONG = c_ulonglong;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PDWORDLONG = *mut DWORDLONG;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type LONG = c_long;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PLONG = *mut LONG;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type LPLONG = *mut LONG;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type LONGLONG = c_longlong;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PLONGLONG = *mut LONGLONG;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type ULONG = c_ulong;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PULONG = *mut ULONG;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type ULONGLONG = c_ulonglong;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PULONGLONG = *mut ULONGLONG;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type LONG32 = c_int;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PLONG32 = *mut LONG32;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type ULONG32 = c_uint;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PULONG32 = *mut LONG32;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type LONG64 = c_longlong;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PLONG64 = *mut LONG64;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type ULONG64 = c_ulonglong;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type PULONG64 = *mut ULONG64;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HFILE = c_int;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HBITMAP = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HBRUSH = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HCOLORSPACE = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HCONV = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HCONVLIST = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HICON = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HDC = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HDDEDATA = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HDESK = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HDROP = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HDWP = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HENFMETAFILE = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HFONT = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HGDIOBJECT = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HGLOBAL = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HHOOK = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HCURSOR = HICON;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HINSTANCE = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HKEY = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HKL = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HLOCAL = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HMENU = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HMETAFILE = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HMODULE = HINSTANCE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HMONITOR = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HPALETTE = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HPEN = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HRESULT = LONG;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HRGN = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HRSRC = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HSZ = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HWINSTA = HANDLE;

#[unstable(feature = "raw_windows_types", reason = "Recently added")]
pub type HWND = HANDLE;