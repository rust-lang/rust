/*
Module: ctypes

Definitions useful for C interop
*/

/*
FIXME: Add a test that uses some native code to verify these sizes,
which are not obviously correct for all potential platforms.
*/

// PORT adapt to architecture

/*
Type: c_int

A signed integer with the same size as a C `int`
*/
type c_int = i32;

/*
Type: c_uint

An unsigned integer with the same size as a C `unsigned int`
*/
type c_uint = u32;

/*
Type: long

A signed integer with the same size as a C `long`
*/
type long = int;

/*
Type: unsigned

An unsigned integer with the same size as a C `unsigned int`
*/
type unsigned = u32;

/*
Type: ulong

An unsigned integer with the same size as a C `unsigned long`
*/
type ulong = uint;

/*
Type: intptr_t

A signed integer with the same size as a pointer. This is
guaranteed to always be the same type as a Rust `int`
*/
type intptr_t = uint; // FIXME: int

/*
Type: uintptr_t

An unsigned integer with the same size as a pointer. This is
guaranteed to always be the same type as a Rust `uint`.
*/
type uintptr_t = uint;
type uint32_t = u32;

/*
Type: void

A type, a pointer to which can be used as C `void *`

Note that this does not directly correspond to the C `void` type,
which is an incomplete type. Using pointers to this type
when interoperating with C void pointers can help in documentation.
*/
type void = int;

/*
Type: c_float

A float value with the same size as a C `float`
*/
type c_float = f32;

/*
Type: c_float

A float value with the same size as a C `double`
*/
type c_double = f64;

/*
Type: size_t

An unsigned integer corresponding to the C `size_t`
*/
type size_t = uint;

/*
Type: ssize_t

A signed integer correpsonding to the C `ssize_t`
*/
type ssize_t = int;

/*
Type: off_t

An unsigned integer corresponding to the C `off_t`
*/
type off_t = uint;

/*
Type: fd_t

A type that can be used for C file descriptors
*/
type fd_t = i32;      // not actually a C type, but should be.

/*
Type: pid_t

A type for representing process ID's, corresponding to C `pid_t`
*/
type pid_t = i32;

// enum is implementation-defined, but is 32-bits in practice
/*
Type: enum

An unsigned integer with the same size as a C enum
*/
type enum = u32;

