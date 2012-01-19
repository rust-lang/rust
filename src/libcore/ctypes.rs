#[doc = "Definitions useful for C interop"];

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

#[doc(
  brief = "A signed integer with the same size as a C `int`."
)]
type c_int = i32;

#[doc(
  brief = "An unsigned integer with the same size as a C `unsigned int`."
)]
type c_uint = u32;

#[doc(
  brief = "A signed integer with the same size as a C `long`."
)]
type long = int;

#[doc(
  brief = "A signed integer with the same size as a C `long long`."
)]
type longlong = i64;

#[doc(
  brief = "A signed integer with the same size as a C `unsigned int`."
)]
type unsigned = u32;

#[doc(
  brief = "A signed integer with the same size as a C `unsigned long`."
)]
type ulong = uint;

#[doc(
  brief = "A signed integer with the same size as a C `unsigned long long`."
)]
type ulonglong = u64;

#[doc(
  brief = "A signed integer with the same size as a pointer. \
           This is guaranteed to always be the same type as a \
            Rust `int`."
)]
type intptr_t = uint; // FIXME: int

#[doc(
  brief = "An unsigned integer with the same size as a pointer. \
           This is guaranteed to always be the same type as a Rust \
           `uint`."
)]
type uintptr_t = uint;
type uint32_t = u32;

#[doc(
  brief = "A type, a pointer to which can be used as C `void *`.",
  desc = "The void type cannot be constructed or destructured, \
         but using pointers to this type when interoperating \
         with C void pointers can help in documentation."
)]
enum void {
    // Making the only variant reference itself makes it impossible to
    // construct. Not exporting it makes it impossible to destructure.
    void_private(@void);
    // FIXME: #881
    void_private2(@void);
}

#[doc(
  brief = "A float value with the same size as a C `float`."
)]
type c_float = f32;

#[doc(
  brief = "A float value with the same size as a C `double`."
)]
type c_double = f64;

#[doc(
  brief = "An unsigned integer corresponding to the C `size_t`."
)]
type size_t = uint;

#[doc(
  brief = "A signed integer corresponding to the C `ssize_t`."
)]
type ssize_t = int;

#[doc(
  brief = "An unsigned integer corresponding to the C `off_t`."
)]
type off_t = uint;

#[doc(
  brief = "A type that can be used for C file descriptors."
)]
type fd_t = i32;      // not actually a C type, but should be.

#[doc(
  brief = "A type for representing process ID's, corresponding to C `pid_t`."
)]
type pid_t = i32;

#[doc(
  brief = "An unsigned integer with the same size as a C enum. \
           enum is implementation-defined, but is 32-bits in \
           practice"
)]
type enum = u32;

