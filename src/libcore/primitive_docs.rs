#[doc(primitive = "bool")]
#[doc(alias = "true")]
#[doc(alias = "false")]
//
/// The boolean type.
///
/// *[See also the primitive documentation in `std`](../std/primitive.bool.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_bool { }

#[doc(primitive = "never")]
#[doc(alias = "!")]
//
/// The `!` type, also called "never".
///
/// *[See also the primitive documentation in `std`](../std/primitive.never.html).*
#[unstable(feature = "never_type", issue = "35121")]
mod prim_never { }

#[doc(primitive = "char")]
//
/// A character type.
///
/// *[See also the primitive documentation in `std`](../std/primitive.char.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_char { }

#[doc(primitive = "unit")]
//
/// The `()` type, sometimes called "unit" or "nil".
///
/// *[See also the primitive documentation in `std`](../std/primitive.unit.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_unit { }

#[doc(primitive = "pointer")]
//
/// Raw, unsafe pointers, `*const T`, and `*mut T`.
///
/// *[See also the `ptr` module](ptr/index.html) and
/// [the primitive documentation in `std`](../std/primitive.pointer.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_pointer { }

#[doc(primitive = "array")]
//
/// A fixed-size array, denoted `[T; N]`, for the element type, `T`, and the
/// non-negative compile-time constant size, `N`.
///
/// *[See also the primitive documentation in `std`](../std/primitive.array.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_array { }

#[doc(primitive = "slice")]
#[doc(alias = "[")]
#[doc(alias = "]")]
#[doc(alias = "[]")]
/// A dynamically-sized view into a contiguous sequence, `[T]`.
///
/// *[See also the `slice` module](slice/index.html) and
/// [the primitive documentation in `std`](../std/primitive.slice.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_slice { }

#[doc(primitive = "str")]
//
/// String slices.
///
/// *[See also the `str` module](str/index.html) and
/// [the primitive documentation in `std`](../std/primitive.str.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_str { }

#[doc(primitive = "tuple")]
#[doc(alias = "(")]
#[doc(alias = ")")]
#[doc(alias = "()")]
//
/// A finite heterogeneous sequence, `(T, U, ..)`.
///
/// *[See also the primitive documentation in `std`](../std/primitive.tuple.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_tuple { }

#[doc(primitive = "f32")]
/// The 32-bit floating point type.
///
/// *[See also the `f32` module](f32/index.html) and
/// [the primitive documentation in `std`](../std/primitive.f32.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_f32 { }

#[doc(primitive = "f64")]
//
/// The 64-bit floating point type.
///
/// *[See also the `f64` module](f64/index.html) and
/// [the primitive documentation in `std`](../std/primitive.f64.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_f64 { }

#[doc(primitive = "i8")]
//
/// The 8-bit signed integer type.
///
/// *[See also the `i8` module](i8/index.html) and
/// [the primitive documentation in `std`](../std/primitive.i8.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_i8 { }

#[doc(primitive = "i16")]
//
/// The 16-bit signed integer type.
///
/// *[See also the `i16` module](i16/index.html) and
/// [the primitive documentation in `std`](../std/primitive.i16.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_i16 { }

#[doc(primitive = "i32")]
//
/// The 32-bit signed integer type.
///
/// *[See also the `i32` module](i32/index.html) and
/// [the primitive documentation in `std`](../std/primitive.i32.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_i32 { }

#[doc(primitive = "i64")]
//
/// The 64-bit signed integer type.
///
/// *[See also the `i64` module](i64/index.html) and
/// [the primitive documentation in `std`](../std/primitive.i64.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_i64 { }

#[doc(primitive = "i128")]
//
/// The 128-bit signed integer type.
///
/// *[See also the `i128` module](i128/index.html) and
/// [the primitive documentation in `std`](../std/primitive.i128.html).*
#[stable(feature = "i128", since="1.26.0")]
mod prim_i128 { }

#[doc(primitive = "u8")]
//
/// The 8-bit unsigned integer type.
///
/// *[See also the `u8` module](u8/index.html) and
/// [the primitive documentation in `std`](../std/primitive.u8.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_u8 { }

#[doc(primitive = "u16")]
//
/// The 16-bit unsigned integer type.
///
/// *[See also the `u16` module](u16/index.html) and
/// [the primitive documentation in `std`](../std/primitive.u16.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_u16 { }

#[doc(primitive = "u32")]
//
/// The 32-bit unsigned integer type.
///
/// *[See also the `u32` module](u32/index.html) and
/// [the primitive documentation in `std`](../std/primitive.u32.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_u32 { }

#[doc(primitive = "u64")]
//
/// The 64-bit unsigned integer type.
///
/// *[See also the `u64` module](u64/index.html) and
/// [the primitive documentation in `std`](../std/primitive.u64.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_u64 { }

#[doc(primitive = "u128")]
//
/// The 128-bit unsigned integer type.
///
/// *[See also the `u128` module](u128/index.html) and
/// [the primitive documentation in `std`](../std/primitive.u128.html).*
#[stable(feature = "i128", since="1.26.0")]
mod prim_u128 { }

#[doc(primitive = "isize")]
//
/// The pointer-sized signed integer type.
///
/// *[See also the `isize` module](isize/index.html) and
/// [the primitive documentation in `std`](../std/primitive.isize.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_isize { }

#[doc(primitive = "usize")]
//
/// The pointer-sized unsigned integer type.
///
/// *[See also the `usize` module](usize/index.html) and
/// [the primitive documentation in `std`](../std/primitive.usize.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_usize { }

#[doc(primitive = "reference")]
#[doc(alias = "&")]
//
/// References, both shared and mutable.
///
/// *[See also the primitive documentation in `std`](../std/primitive.reference.html).*
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_ref { }

#[doc(primitive = "fn")]
//
/// Function pointers, like `fn(usize) -> bool`.
///
/// *See also the traits [`Fn`], [`FnMut`], and [`FnOnce`], and
/// [the primitive documentation in `std`](../std/primitive.fn.html).*
///
/// [`Fn`]: ops/trait.Fn.html
/// [`FnMut`]: ops/trait.FnMut.html
/// [`FnOnce`]: ops/trait.FnOnce.html
#[stable(feature = "rust1", since = "1.0.0")]
mod prim_fn { }
