/*
Module: ctypes

Definitions useful for C interop
*/

type c_int = i32;
type c_uint = u32;

type void = int; // Not really the same as C
type long = int;
type unsigned = u32;
type ulong = uint;

type intptr_t = uint;
type uintptr_t = uint;
type uint32_t = u32;

// machine type equivalents of rust int, uint, float

#[cfg(target_arch="x86")]
type m_int = i32;
#[cfg(target_arch="x86_64")]
type m_int = i64;

#[cfg(target_arch="x86")]
type m_uint = u32;
#[cfg(target_arch="x86_64")]
type m_uint = u64;

// This *must* match with "import m_float = fXX" in std::math per arch
type m_float = f64;

type size_t = uint;
type ssize_t = int;
type off_t = uint;

type fd_t = i32;      // not actually a C type, but should be.
type pid_t = i32;

// enum is implementation-defined, but is 32-bits in practice
type enum = u32;
