% Casts

Casts are a superset of coercions: every coercion can be explicitly
invoked via a cast. However some conversions require a cast.
While coercions are pervasive and largely harmless, these "true casts"
are rare and potentially dangerous. As such, casts must be explicitly invoked
using the `as` keyword: `expr as Type`.

True casts generally revolve around raw pointers and the primitive numeric
types. Even though they're dangerous, these casts are infallible at runtime.
If a cast triggers some subtle corner case no indication will be given that
this occurred. The cast will simply succeed. That said, casts must be valid
at the type level, or else they will be prevented statically. For instance,
`7u8 as bool` will not compile.

That said, casts aren't `unsafe` because they generally can't violate memory
safety *on their own*. For instance, converting an integer to a raw pointer can
very easily lead to terrible things. However the act of creating the pointer
itself is safe, because actually using a raw pointer is already marked as
`unsafe`.

Here's an exhaustive list of all the true casts. For brevity, we will use `*`
to denote either a `*const` or `*mut`, and `integer` to denote any integral
primitive:

 * `*T as *U` where `T, U: Sized`
 * `*T as *U` TODO: explain unsized situation
 * `*T as integer`
 * `integer as *T`
 * `number as number`
 * `C-like-enum as integer`
 * `bool as integer`
 * `char as integer`
 * `u8 as char`
 * `&[T; n] as *const T`
 * `fn as *T` where `T: Sized`
 * `fn as integer`

Note that lengths are not adjusted when casting raw slices -
`*const [u16] as *const [u8]` creates a slice that only includes
half of the original memory.

Casting is not transitive, that is, even if `e as U1 as U2` is a valid
expression, `e as U2` is not necessarily so.

For numeric casts, there are quite a few cases to consider:

* casting between two integers of the same size (e.g. i32 -> u32) is a no-op
* casting from a larger integer to a smaller integer (e.g. u32 -> u8) will
  truncate
* casting from a smaller integer to a larger integer (e.g. u8 -> u32) will
    * zero-extend if the source is unsigned
    * sign-extend if the source is signed
* casting from a float to an integer will round the float towards zero
    * **[NOTE: currently this will cause Undefined Behavior if the rounded
      value cannot be represented by the target integer type][float-int]**.
      This includes Inf and NaN. This is a bug and will be fixed.
* casting from an integer to float will produce the floating point
  representation of the integer, rounded if necessary (rounding strategy
  unspecified)
* casting from an f32 to an f64 is perfect and lossless
* casting from an f64 to an f32 will produce the closest possible value
  (rounding strategy unspecified)
    * **[NOTE: currently this will cause Undefined Behavior if the value
      is finite but larger or smaller than the largest or smallest finite
      value representable by f32][float-float]**. This is a bug and will
      be fixed.


[float-int]: https://github.com/rust-lang/rust/issues/10184
[float-float]: https://github.com/rust-lang/rust/issues/15536
