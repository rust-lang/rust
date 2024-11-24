//! SIMD compiler intrinsics.
//!
//! In this module, a "vector" is any `repr(simd)` type.

extern "rust-intrinsic" {
    /// Inserts an element into a vector, returning the updated vector.
    ///
    /// `T` must be a vector with element type `U`.
    ///
    /// # Safety
    ///
    /// `idx` must be in-bounds of the vector.
    #[rustc_nounwind]
    pub fn simd_insert<T, U>(x: T, idx: u32, val: U) -> T;

    /// Extracts an element from a vector.
    ///
    /// `T` must be a vector with element type `U`.
    ///
    /// # Safety
    ///
    /// `idx` must be in-bounds of the vector.
    #[rustc_nounwind]
    pub fn simd_extract<T, U>(x: T, idx: u32) -> U;

    /// Adds two simd vectors elementwise.
    ///
    /// `T` must be a vector of integer or floating point primitive types.
    #[rustc_nounwind]
    pub fn simd_add<T>(x: T, y: T) -> T;

    /// Subtracts `rhs` from `lhs` elementwise.
    ///
    /// `T` must be a vector of integer or floating point primitive types.
    #[rustc_nounwind]
    pub fn simd_sub<T>(lhs: T, rhs: T) -> T;

    /// Multiplies two simd vectors elementwise.
    ///
    /// `T` must be a vector of integer or floating point primitive types.
    #[rustc_nounwind]
    pub fn simd_mul<T>(x: T, y: T) -> T;

    /// Divides `lhs` by `rhs` elementwise.
    ///
    /// `T` must be a vector of integer or floating point primitive types.
    ///
    /// # Safety
    /// For integers, `rhs` must not contain any zero elements.
    /// Additionally for signed integers, `<int>::MIN / -1` is undefined behavior.
    #[rustc_nounwind]
    pub fn simd_div<T>(lhs: T, rhs: T) -> T;

    /// Returns remainder of two vectors elementwise.
    ///
    /// `T` must be a vector of integer or floating point primitive types.
    ///
    /// # Safety
    /// For integers, `rhs` must not contain any zero elements.
    /// Additionally for signed integers, `<int>::MIN / -1` is undefined behavior.
    #[rustc_nounwind]
    pub fn simd_rem<T>(lhs: T, rhs: T) -> T;

    /// Shifts vector left elementwise, with UB on overflow.
    ///
    /// Shifts `lhs` left by `rhs`, shifting in sign bits for signed types.
    ///
    /// `T` must be a vector of integer primitive types.
    ///
    /// # Safety
    ///
    /// Each element of `rhs` must be less than `<int>::BITS`.
    #[rustc_nounwind]
    pub fn simd_shl<T>(lhs: T, rhs: T) -> T;

    /// Shifts vector right elementwise, with UB on overflow.
    ///
    /// `T` must be a vector of integer primitive types.
    ///
    /// Shifts `lhs` right by `rhs`, shifting in sign bits for signed types.
    ///
    /// # Safety
    ///
    /// Each element of `rhs` must be less than `<int>::BITS`.
    #[rustc_nounwind]
    pub fn simd_shr<T>(lhs: T, rhs: T) -> T;

    /// "Ands" vectors elementwise.
    ///
    /// `T` must be a vector of integer primitive types.
    #[rustc_nounwind]
    pub fn simd_and<T>(x: T, y: T) -> T;

    /// "Ors" vectors elementwise.
    ///
    /// `T` must be a vector of integer primitive types.
    #[rustc_nounwind]
    pub fn simd_or<T>(x: T, y: T) -> T;

    /// "Exclusive ors" vectors elementwise.
    ///
    /// `T` must be a vector of integer primitive types.
    #[rustc_nounwind]
    pub fn simd_xor<T>(x: T, y: T) -> T;

    /// Numerically casts a vector, elementwise.
    ///
    /// `T` and `U` must be vectors of integer or floating point primitive types, and must have the
    /// same length.
    ///
    /// When casting floats to integers, the result is truncated. Out-of-bounds result lead to UB.
    /// When casting integers to floats, the result is rounded.
    /// Otherwise, truncates or extends the value, maintaining the sign for signed integers.
    ///
    /// # Safety
    /// Casting from integer types is always safe.
    /// Casting between two float types is also always safe.
    ///
    /// Casting floats to integers truncates, following the same rules as `to_int_unchecked`.
    /// Specifically, each element must:
    /// * Not be `NaN`
    /// * Not be infinite
    /// * Be representable in the return type, after truncating off its fractional part
    #[rustc_nounwind]
    pub fn simd_cast<T, U>(x: T) -> U;

    /// Numerically casts a vector, elementwise.
    ///
    /// `T` and `U` be a vectors of integer or floating point primitive types, and must have the
    /// same length.
    ///
    /// Like `simd_cast`, but saturates float-to-integer conversions (NaN becomes 0).
    /// This matches regular `as` and is always safe.
    ///
    /// When casting floats to integers, the result is truncated.
    /// When casting integers to floats, the result is rounded.
    /// Otherwise, truncates or extends the value, maintaining the sign for signed integers.
    #[rustc_nounwind]
    pub fn simd_as<T, U>(x: T) -> U;

    /// Negates a vector elementwise.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// Rust panics for `-<int>::Min` due to overflow, but it is not UB with this intrinsic.
    #[rustc_nounwind]
    pub fn simd_neg<T>(x: T) -> T;

    /// Returns absolute value of a vector, elementwise.
    ///
    /// `T` must be a vector of floating-point primitive types.
    #[rustc_nounwind]
    pub fn simd_fabs<T>(x: T) -> T;

    /// Returns the minimum of two vectors, elementwise.
    ///
    /// `T` must be a vector of floating-point primitive types.
    ///
    /// Follows IEEE-754 `minNum` semantics.
    #[rustc_nounwind]
    pub fn simd_fmin<T>(x: T, y: T) -> T;

    /// Returns the maximum of two vectors, elementwise.
    ///
    /// `T` must be a vector of floating-point primitive types.
    ///
    /// Follows IEEE-754 `maxNum` semantics.
    #[rustc_nounwind]
    pub fn simd_fmax<T>(x: T, y: T) -> T;

    /// Tests elementwise equality of two vectors.
    ///
    /// `T` must be a vector of floating-point primitive types.
    ///
    /// `U` must be a vector of integers with the same number of elements and element size as `T`.
    ///
    /// Returns `0` for false and `!0` for true.
    #[rustc_nounwind]
    pub fn simd_eq<T, U>(x: T, y: T) -> U;

    /// Tests elementwise inequality equality of two vectors.
    ///
    /// `T` must be a vector of floating-point primitive types.
    ///
    /// `U` must be a vector of integers with the same number of elements and element size as `T`.
    ///
    /// Returns `0` for false and `!0` for true.
    #[rustc_nounwind]
    pub fn simd_ne<T, U>(x: T, y: T) -> U;

    /// Tests if `x` is less than `y`, elementwise.
    ///
    /// `T` must be a vector of floating-point primitive types.
    ///
    /// `U` must be a vector of integers with the same number of elements and element size as `T`.
    ///
    /// Returns `0` for false and `!0` for true.
    #[rustc_nounwind]
    pub fn simd_lt<T, U>(x: T, y: T) -> U;

    /// Tests if `x` is less than or equal to `y`, elementwise.
    ///
    /// `T` must be a vector of floating-point primitive types.
    ///
    /// `U` must be a vector of integers with the same number of elements and element size as `T`.
    ///
    /// Returns `0` for false and `!0` for true.
    #[rustc_nounwind]
    pub fn simd_le<T, U>(x: T, y: T) -> U;

    /// Tests if `x` is greater than `y`, elementwise.
    ///
    /// `T` must be a vector of floating-point primitive types.
    ///
    /// `U` must be a vector of integers with the same number of elements and element size as `T`.
    ///
    /// Returns `0` for false and `!0` for true.
    #[rustc_nounwind]
    pub fn simd_gt<T, U>(x: T, y: T) -> U;

    /// Tests if `x` is greater than or equal to `y`, elementwise.
    ///
    /// `T` must be a vector of floating-point primitive types.
    ///
    /// `U` must be a vector of integers with the same number of elements and element size as `T`.
    ///
    /// Returns `0` for false and `!0` for true.
    #[rustc_nounwind]
    pub fn simd_ge<T, U>(x: T, y: T) -> U;

    /// Shuffles two vectors by const indices.
    ///
    /// `T` must be a vector.
    ///
    /// `U` must be a **const** vector of `u32`s. This means it must either refer to a named
    /// const or be given as an inline const expression (`const { ... }`).
    ///
    /// `V` must be a vector with the same element type as `T` and the same length as `U`.
    ///
    /// Returns a new vector such that element `i` is selected from `xy[idx[i]]`, where `xy`
    /// is the concatenation of `x` and `y`. It is a compile-time error if `idx[i]` is out-of-bounds
    /// of `xy`.
    #[rustc_nounwind]
    pub fn simd_shuffle<T, U, V>(x: T, y: T, idx: U) -> V;

    /// Reads a vector of pointers.
    ///
    /// `T` must be a vector.
    ///
    /// `U` must be a vector of pointers to the element type of `T`, with the same length as `T`.
    ///
    /// `V` must be a vector of integers with the same length as `T` (but any element size).
    ///
    /// For each pointer in `ptr`, if the corresponding value in `mask` is `!0`, read the pointer.
    /// Otherwise if the corresponding value in `mask` is `0`, return the corresponding value from
    /// `val`.
    ///
    /// # Safety
    /// Unmasked values in `T` must be readable as if by `<ptr>::read` (e.g. aligned to the element
    /// type).
    ///
    /// `mask` must only contain `0` or `!0` values.
    #[rustc_nounwind]
    pub fn simd_gather<T, U, V>(val: T, ptr: U, mask: V) -> T;

    /// Writes to a vector of pointers.
    ///
    /// `T` must be a vector.
    ///
    /// `U` must be a vector of pointers to the element type of `T`, with the same length as `T`.
    ///
    /// `V` must be a vector of integers with the same length as `T` (but any element size).
    ///
    /// For each pointer in `ptr`, if the corresponding value in `mask` is `!0`, write the
    /// corresponding value in `val` to the pointer.
    /// Otherwise if the corresponding value in `mask` is `0`, do nothing.
    ///
    /// The stores happen in left-to-right order.
    /// (This is relevant in case two of the stores overlap.)
    ///
    /// # Safety
    /// Unmasked values in `T` must be writeable as if by `<ptr>::write` (e.g. aligned to the element
    /// type).
    ///
    /// `mask` must only contain `0` or `!0` values.
    #[rustc_nounwind]
    pub fn simd_scatter<T, U, V>(val: T, ptr: U, mask: V);

    /// Reads a vector of pointers.
    ///
    /// `T` must be a vector.
    ///
    /// `U` must be a pointer to the element type of `T`
    ///
    /// `V` must be a vector of integers with the same length as `T` (but any element size).
    ///
    /// For each element, if the corresponding value in `mask` is `!0`, read the corresponding
    /// pointer offset from `ptr`.
    /// The first element is loaded from `ptr`, the second from `ptr.wrapping_offset(1)` and so on.
    /// Otherwise if the corresponding value in `mask` is `0`, return the corresponding value from
    /// `val`.
    ///
    /// # Safety
    /// Unmasked values in `T` must be readable as if by `<ptr>::read` (e.g. aligned to the element
    /// type).
    ///
    /// `mask` must only contain `0` or `!0` values.
    #[rustc_nounwind]
    pub fn simd_masked_load<V, U, T>(mask: V, ptr: U, val: T) -> T;

    /// Writes to a vector of pointers.
    ///
    /// `T` must be a vector.
    ///
    /// `U` must be a pointer to the element type of `T`
    ///
    /// `V` must be a vector of integers with the same length as `T` (but any element size).
    ///
    /// For each element, if the corresponding value in `mask` is `!0`, write the corresponding
    /// value in `val` to the pointer offset from `ptr`.
    /// The first element is written to `ptr`, the second to `ptr.wrapping_offset(1)` and so on.
    /// Otherwise if the corresponding value in `mask` is `0`, do nothing.
    ///
    /// # Safety
    /// Unmasked values in `T` must be writeable as if by `<ptr>::write` (e.g. aligned to the element
    /// type).
    ///
    /// `mask` must only contain `0` or `!0` values.
    #[rustc_nounwind]
    pub fn simd_masked_store<V, U, T>(mask: V, ptr: U, val: T);

    /// Adds two simd vectors elementwise, with saturation.
    ///
    /// `T` must be a vector of integer primitive types.
    #[rustc_nounwind]
    pub fn simd_saturating_add<T>(x: T, y: T) -> T;

    /// Subtracts two simd vectors elementwise, with saturation.
    ///
    /// `T` must be a vector of integer primitive types.
    ///
    /// Subtract `rhs` from `lhs`.
    #[rustc_nounwind]
    pub fn simd_saturating_sub<T>(lhs: T, rhs: T) -> T;

    /// Adds elements within a vector from left to right.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    ///
    /// Starting with the value `y`, add the elements of `x` and accumulate.
    #[rustc_nounwind]
    pub fn simd_reduce_add_ordered<T, U>(x: T, y: U) -> U;

    /// Adds elements within a vector in arbitrary order. May also be re-associated with
    /// unordered additions on the inputs/outputs.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    #[rustc_nounwind]
    pub fn simd_reduce_add_unordered<T, U>(x: T) -> U;

    /// Multiplies elements within a vector from left to right.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    ///
    /// Starting with the value `y`, multiply the elements of `x` and accumulate.
    #[rustc_nounwind]
    pub fn simd_reduce_mul_ordered<T, U>(x: T, y: U) -> U;

    /// Multiplies elements within a vector in arbitrary order. May also be re-associated with
    /// unordered additions on the inputs/outputs.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    #[rustc_nounwind]
    pub fn simd_reduce_mul_unordered<T, U>(x: T) -> U;

    /// Checks if all mask values are true.
    ///
    /// `T` must be a vector of integer primitive types.
    ///
    /// # Safety
    /// `x` must contain only `0` or `!0`.
    #[rustc_nounwind]
    pub fn simd_reduce_all<T>(x: T) -> bool;

    /// Checks if any mask value is true.
    ///
    /// `T` must be a vector of integer primitive types.
    ///
    /// # Safety
    /// `x` must contain only `0` or `!0`.
    #[rustc_nounwind]
    pub fn simd_reduce_any<T>(x: T) -> bool;

    /// Returns the maximum element of a vector.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    ///
    /// For floating-point values, uses IEEE-754 `maxNum`.
    #[rustc_nounwind]
    pub fn simd_reduce_max<T, U>(x: T) -> U;

    /// Returns the minimum element of a vector.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    ///
    /// For floating-point values, uses IEEE-754 `minNum`.
    #[rustc_nounwind]
    pub fn simd_reduce_min<T, U>(x: T) -> U;

    /// Logical "ands" all elements together.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    #[rustc_nounwind]
    pub fn simd_reduce_and<T, U>(x: T) -> U;

    /// Logical "ors" all elements together.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    #[rustc_nounwind]
    pub fn simd_reduce_or<T, U>(x: T) -> U;

    /// Logical "exclusive ors" all elements together.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    #[rustc_nounwind]
    pub fn simd_reduce_xor<T, U>(x: T) -> U;

    /// Truncates an integer vector to a bitmask.
    ///
    /// `T` must be an integer vector.
    ///
    /// `U` must be either the smallest unsigned integer with at least as many bits as the length
    /// of `T`, or the smallest array of `u8` with at least as many bits as the length of `T`.
    ///
    /// Each element is truncated to a single bit and packed into the result.
    ///
    /// No matter whether the output is an array or an unsigned integer, it is treated as a single
    /// contiguous list of bits. The bitmask is always packed on the least-significant side of the
    /// output, and padded with 0s in the most-significant bits. The order of the bits depends on
    /// endianness:
    ///
    /// * On little endian, the least significant bit corresponds to the first vector element.
    /// * On big endian, the least significant bit corresponds to the last vector element.
    ///
    /// For example, `[!0, 0, !0, !0]` packs to
    /// - `0b1101u8` or `[0b1101]` on little endian, and
    /// - `0b1011u8` or `[0b1011]` on big endian.
    ///
    /// To consider a larger example,
    /// `[!0, 0, 0, 0, 0, 0, 0, 0, !0, !0, 0, 0, 0, 0, !0, 0]` packs to
    /// - `0b0100001100000001u16` or `[0b00000001, 0b01000011]` on little endian, and
    /// - `0b1000000011000010u16` or `[0b10000000, 0b11000010]` on big endian.
    ///
    /// And finally, a non-power-of-2 example with multiple bytes:
    /// `[!0, !0, 0, !0, 0, 0, !0, 0, !0, 0]` packs to
    /// - `0b0101001011u16` or `[0b01001011, 0b01]` on little endian, and
    /// - `0b1101001010u16` or `[0b11, 0b01001010]` on big endian.
    ///
    /// # Safety
    /// `x` must contain only `0` and `!0`.
    #[rustc_nounwind]
    pub fn simd_bitmask<T, U>(x: T) -> U;

    /// Selects elements from a mask.
    ///
    /// `M` must be an integer vector.
    ///
    /// `T` must be a vector with the same number of elements as `M`.
    ///
    /// For each element, if the corresponding value in `mask` is `!0`, select the element from
    /// `if_true`.  If the corresponding value in `mask` is `0`, select the element from
    /// `if_false`.
    ///
    /// # Safety
    /// `mask` must only contain `0` and `!0`.
    #[rustc_nounwind]
    pub fn simd_select<M, T>(mask: M, if_true: T, if_false: T) -> T;

    /// Selects elements from a bitmask.
    ///
    /// `M` must be an unsigned integer or array of `u8`, matching `simd_bitmask`.
    ///
    /// `T` must be a vector.
    ///
    /// For each element, if the bit in `mask` is `1`, select the element from
    /// `if_true`.  If the corresponding bit in `mask` is `0`, select the element from
    /// `if_false`.
    ///
    /// The bitmask bit order matches `simd_bitmask`.
    ///
    /// # Safety
    /// Padding bits must be all zero.
    #[rustc_nounwind]
    pub fn simd_select_bitmask<M, T>(m: M, yes: T, no: T) -> T;

    /// Calculates the offset from a pointer vector elementwise, potentially
    /// wrapping.
    ///
    /// `T` must be a vector of pointers.
    ///
    /// `U` must be a vector of `isize` or `usize` with the same number of elements as `T`.
    ///
    /// Operates as if by `<ptr>::wrapping_offset`.
    #[rustc_nounwind]
    pub fn simd_arith_offset<T, U>(ptr: T, offset: U) -> T;

    /// Casts a vector of pointers.
    ///
    /// `T` and `U` must be vectors of pointers with the same number of elements.
    #[rustc_nounwind]
    pub fn simd_cast_ptr<T, U>(ptr: T) -> U;

    /// Exposes a vector of pointers as a vector of addresses.
    ///
    /// `T` must be a vector of pointers.
    ///
    /// `U` must be a vector of `usize` with the same length as `T`.
    #[rustc_nounwind]
    pub fn simd_expose_provenance<T, U>(ptr: T) -> U;

    /// Creates a vector of pointers from a vector of addresses.
    ///
    /// `T` must be a vector of `usize`.
    ///
    /// `U` must be a vector of pointers, with the same length as `T`.
    #[rustc_nounwind]
    pub fn simd_with_exposed_provenance<T, U>(addr: T) -> U;

    /// Swaps bytes of each element.
    ///
    /// `T` must be a vector of integers.
    #[rustc_nounwind]
    pub fn simd_bswap<T>(x: T) -> T;

    /// Reverses bits of each element.
    ///
    /// `T` must be a vector of integers.
    #[rustc_nounwind]
    pub fn simd_bitreverse<T>(x: T) -> T;

    /// Counts the leading zeros of each element.
    ///
    /// `T` must be a vector of integers.
    #[rustc_nounwind]
    pub fn simd_ctlz<T>(x: T) -> T;

    /// Counts the number of ones in each element.
    ///
    /// `T` must be a vector of integers.
    #[rustc_nounwind]
    pub fn simd_ctpop<T>(x: T) -> T;

    /// Counts the trailing zeros of each element.
    ///
    /// `T` must be a vector of integers.
    #[rustc_nounwind]
    pub fn simd_cttz<T>(x: T) -> T;

    /// Rounds up each element to the next highest integer-valued float.
    ///
    /// `T` must be a vector of floats.
    #[rustc_nounwind]
    pub fn simd_ceil<T>(x: T) -> T;

    /// Rounds down each element to the next lowest integer-valued float.
    ///
    /// `T` must be a vector of floats.
    #[rustc_nounwind]
    pub fn simd_floor<T>(x: T) -> T;

    /// Rounds each element to the closest integer-valued float.
    /// Ties are resolved by rounding away from 0.
    ///
    /// `T` must be a vector of floats.
    #[rustc_nounwind]
    pub fn simd_round<T>(x: T) -> T;

    /// Returns the integer part of each element as an integer-valued float.
    /// In other words, non-integer values are truncated towards zero.
    ///
    /// `T` must be a vector of floats.
    #[rustc_nounwind]
    pub fn simd_trunc<T>(x: T) -> T;

    /// Takes the square root of each element.
    ///
    /// `T` must be a vector of floats.
    #[rustc_nounwind]
    pub fn simd_fsqrt<T>(x: T) -> T;

    /// Computes `(x*y) + z` for each element, but without any intermediate rounding.
    ///
    /// `T` must be a vector of floats.
    #[rustc_nounwind]
    pub fn simd_fma<T>(x: T, y: T, z: T) -> T;

    /// Computes `(x*y) + z` for each element, non-deterministically executing either
    /// a fused multiply-add or two operations with rounding of the intermediate result.
    ///
    /// The operation is fused if the code generator determines that target instruction
    /// set has support for a fused operation, and that the fused operation is more efficient
    /// than the equivalent, separate pair of mul and add instructions. It is unspecified
    /// whether or not a fused operation is selected, and that may depend on optimization
    /// level and context, for example.
    ///
    /// `T` must be a vector of floats.
    #[cfg(not(bootstrap))]
    #[rustc_nounwind]
    pub fn simd_relaxed_fma<T>(x: T, y: T, z: T) -> T;

    // Computes the sine of each element.
    ///
    /// `T` must be a vector of floats.
    #[rustc_nounwind]
    pub fn simd_fsin<T>(a: T) -> T;

    // Computes the cosine of each element.
    ///
    /// `T` must be a vector of floats.
    #[rustc_nounwind]
    pub fn simd_fcos<T>(a: T) -> T;

    // Computes the exponential function of each element.
    ///
    /// `T` must be a vector of floats.
    #[rustc_nounwind]
    pub fn simd_fexp<T>(a: T) -> T;

    // Computes 2 raised to the power of each element.
    ///
    /// `T` must be a vector of floats.
    #[rustc_nounwind]
    pub fn simd_fexp2<T>(a: T) -> T;

    // Computes the base 10 logarithm of each element.
    ///
    /// `T` must be a vector of floats.
    #[rustc_nounwind]
    pub fn simd_flog10<T>(a: T) -> T;

    // Computes the base 2 logarithm of each element.
    ///
    /// `T` must be a vector of floats.
    #[rustc_nounwind]
    pub fn simd_flog2<T>(a: T) -> T;

    // Computes the natural logarithm of each element.
    ///
    /// `T` must be a vector of floats.
    #[rustc_nounwind]
    pub fn simd_flog<T>(a: T) -> T;
}
