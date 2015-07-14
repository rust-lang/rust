% Casts

Casts are a superset of coercions: every coercion can be explicitly invoked via
a cast, but some conversions *require* a cast. These "true casts" are generally
regarded as dangerous or problematic actions. True casts revolve around raw
pointers and the primitive numeric types. True casts aren't checked.

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

where `&.T` and `*T` are references of either mutability,
and where unsize_kind(`T`) is the kind of the unsize info
in `T` - the vtable for a trait definition (e.g. `fmt::Display` or
`Iterator`, not `Iterator<Item=u8>`) or a length (or `()` if `T: Sized`).

Note that lengths are not adjusted when casting raw slices -
`T: *const [u16] as *const [u8]` creates a slice that only includes
half of the original memory.

Casting is not transitive, that is, even if `e as U1 as U2` is a valid
expression, `e as U2` is not necessarily so (in fact it will only be valid if
`U1` coerces to `U2`).

For numeric casts, there are quite a few cases to consider:

* casting between two integers of the same size (e.g. i32 -> u32) is a no-op
* casting from a larger integer to a smaller integer (e.g. u32 -> u8) will
  truncate
* casting from a smaller integer to a larger integer (e.g. u8 -> u32) will
    * zero-extend if the source is unsigned
    * sign-extend if the source is signed
* casting from a float to an integer will round the float towards zero
    * **NOTE: currently this will cause Undefined Behaviour if the rounded
      value cannot be represented by the target integer type**. This includes
      Inf and NaN. This is a bug and will be fixed.
* casting from an integer to float will produce the floating point
  representation of the integer, rounded if necessary (rounding strategy
  unspecified)
* casting from an f32 to an f64 is perfect and lossless
* casting from an f64 to an f32 will produce the closest possible value
  (rounding strategy unspecified)
    * **NOTE: currently this will cause Undefined Behaviour if the value
      is finite but larger or smaller than the largest or smallest finite
      value representable by f32**. This is a bug and will be fixed.
