/*

Module: mtypes

Machine type equivalents of rust int, uint, float, and complex.

Types useful for interop with C when writing bindings that exist
for different types (float, f32, f64, ...; cf float.rs for an example)
*/

export m_int, m_uint, m_float;

// PORT Change this when porting to a new architecture

/*
Type: m_int

Machine type equivalent of an int
*/
#[cfg(target_arch="x86")]
type m_int = i32;
#[cfg(target_arch="x86_64")]
type m_int = i64;

// PORT Change this when porting to a new architecture

/*
Type: m_uint

Machine type equivalent of a uint
*/
#[cfg(target_arch="x86")]
type m_uint = u32;
#[cfg(target_arch="x86_64")]
type m_uint = u64;

// PORT *must* match with "import m_float = fXX" in std::math per arch

/*
Type: m_float

Machine type equivalent of a float
*/
type m_float = f64;

// PORT  *must* match "import m_complex = ..." in std::complex per arch

/*
FIXME Type m_complex

Machine type representing a complex value that uses floats for
both the real and the imaginary part.
*/
// type m_complex = complex_c64::t;

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
