/*
Module: ctypes

Definitions useful for C interop
*/

type c_int = i32;
type long = int;
type unsigned = u32;
type ulong = uint;

type intptr_t = uint;
type uintptr_t = uint;
type uint32_t = u32;

type size_t = uint;
type ssize_t = int;
type off_t = uint;

type fd_t = i32;      // not actually a C type, but should be.
type pid_t = i32;
