/*
Module: ctypes

Definitions useful for C interop
*/

/*
FIXME: Add a test that uses some native code to verify these sizes,
which are not obviously correct for all potential platforms.
*/

export c_int, c_uint, long, longlong, unsigned, ulong, ulonglong;
export intptr_t, uintptr_t;
export uint32_t;
export void;
export c_float, c_double;
export size_t, ssize_t;
export off_t, fd_t, pid_t;
export enum;

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
Type: longlong

A signed integer with the same size as a C `long long`
*/
type longlong = i64;

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
Type: ulonglong

An unsigned integer with the same size as a C `unsigned long long`
*/
type ulonglong = u64;

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

A type, a pointer to which can be used as C `void *`.

The void type cannot be constructed or destructured, but using
pointers to this type when interoperating with C void pointers can
help in documentation.
*/
tag void {
    // Making the only variant reference itself makes it impossible to
    // construct. Not exporting it makes it impossible to destructure.
    void_private(@void);
    // FIXME: #881
    void_private2(@void);
}

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

