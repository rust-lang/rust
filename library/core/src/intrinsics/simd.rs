//! SIMD compiler intrinsics.
//!
//! In this module, a "vector" is any `repr(simd)` type.

extern "platform-intrinsic" {
    /// Add two simd vectors elementwise.
    ///
    /// `T` must be a vector of integer or floating point primitive types.
    pub fn simd_add<T>(x: T, y: T) -> T;

    /// Subtract `rhs` from `lhs` elementwise.
    ///
    /// `T` must be a vector of integer or floating point primitive types.
    pub fn simd_sub<T>(lhs: T, rhs: T) -> T;

    /// Multiply two simd vectors elementwise.
    ///
    /// `T` must be a vector of integer or floating point primitive types.
    pub fn simd_mul<T>(x: T, y: T) -> T;

    /// Divide `lhs` by `rhs` elementwise.
    ///
    /// `T` must be a vector of integer or floating point primitive types.
    ///
    /// # Safety
    /// For integers, `rhs` must not contain any zero elements.
    /// Additionally for signed integers, `<int>::MIN / -1` is undefined behavior.
    pub fn simd_div<T>(lhs: T, rhs: T) -> T;

    /// Remainder of two vectors elementwise
    ///
    /// `T` must be a vector of integer or floating point primitive types.
    ///
    /// # Safety
    /// For integers, `rhs` must not contain any zero elements.
    /// Additionally for signed integers, `<int>::MIN / -1` is undefined behavior.
    pub fn simd_rem<T>(lhs: T, rhs: T) -> T;

    /// Elementwise vector left shift.
    ///
    /// Shift `lhs` left by `rhs`, shifting in sign bits for signed types.
    ///
    /// `T` must be a vector of integer primitive types.
    ///
    /// # Safety
    ///
    /// Each element of `rhs` must be less than `<int>::BITS`.
    pub fn simd_shl<T>(lhs: T, rhs: T) -> T;

    /// Elementwise vector right shift.
    ///
    /// Shift `lhs` right by `rhs`, shifting in sign bits for signed types.
    ///
    /// `T` must be a vector of integer primitive types.
    ///
    /// # Safety
    ///
    /// Each element of `rhs` must be less than `<int>::BITS`.
    pub fn simd_shr<T>(lhs: T, rhs: T) -> T;

    /// Elementwise vector "and".
    ///
    /// `T` must be a vector of integer primitive types.
    pub fn simd_and<T>(x: T, y: T) -> T;

    /// Elementwise vector "or".
    ///
    /// `T` must be a vector of integer primitive types.
    pub fn simd_or<T>(x: T, y: T) -> T;

    /// Elementwise vector "exclusive or".
    ///
    /// `T` must be a vector of integer primitive types.
    pub fn simd_xor<T>(x: T, y: T) -> T;

    /// Numerically cast a vector, elementwise.
    ///
    /// When casting floats to integers, the result is truncated.
    /// When casting integers to floats, the result is rounded.
    /// Otherwise, truncates or extends the value, maintaining the sign for signed integers.
    ///
    /// `T` and `U` be a vectors of integer or floating point primitive types, and must have the
    /// same length.
    ///
    /// # Safety
    /// Casting floats to integers truncates, but the truncated value must fit in the target type.
    pub fn simd_cast<T, U>(x: T) -> U;

    /// Numerically cast a vector, elementwise.
    ///
    /// Like `simd_cast`, but saturates float-to-integer conversions.
    /// This matches regular `as` and is always safe.
    ///
    /// When casting floats to integers, the result is truncated.
    /// When casting integers to floats, the result is rounded.
    /// Otherwise, truncates or extends the value, maintaining the sign for signed integers.
    ///
    /// `T` and `U` be a vectors of integer or floating point primitive types, and must have the
    /// same length.
    pub fn simd_as<T, U>(x: T) -> U;

    /// Elementwise negation of a vector.
    ///
    /// Rust panics for `-<int>::Min` due to overflow, but it is not UB with this intrinsic.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    pub fn simd_neg<T>(x: T) -> T;

    /// Elementwise absolute value of a vector.
    ///
    /// `T` must be a vector of floating-point primitive types.
    pub fn simd_fabs<T>(x: T) -> T;

    /// Elementwise minimum of a vector.
    ///
    /// Follows IEEE-754 `minNum` semantics.
    ///
    /// `T` must be a vector of floating-point primitive types.
    pub fn simd_fmin<T>(x: T, y: T) -> T;

    /// Elementwise maximum of a vector.
    ///
    /// Follows IEEE-754 `maxNum` semantics.
    ///
    /// `T` must be a vector of floating-point primitive types.
    pub fn simd_fmax<T>(x: T, y: T) -> T;

    /// Tests elementwise equality of two vectors.
    ///
    /// Returns `0` for false and `!0` for true.
    ///
    /// `T` must be a vector of floating-point primitive types.
    /// `U` must be a vector of integers with the same number of elements and element size as `T`.
    pub fn simd_eq<T, U>(x: T, y: T) -> U;

    /// Tests elementwise inequality equality of two vectors.
    ///
    /// Returns `0` for false and `!0` for true.
    ///
    /// `T` must be a vector of floating-point primitive types.
    ///
    /// `U` must be a vector of integers with the same number of elements and element size as `T`.
    pub fn simd_ne<T, U>(x: T, y: T) -> U;

    /// Tests if `x` is less than `y`, elementwise.
    ///
    /// Returns `0` for false and `!0` for true.
    ///
    /// `T` must be a vector of floating-point primitive types.
    ///
    /// `U` must be a vector of integers with the same number of elements and element size as `T`.
    pub fn simd_lt<T, U>(x: T, y: T) -> U;

    /// Tests if `x` is less than or equal to `y`, elementwise.
    ///
    /// Returns `0` for false and `!0` for true.
    ///
    /// `T` must be a vector of floating-point primitive types.
    ///
    /// `U` must be a vector of integers with the same number of elements and element size as `T`.
    pub fn simd_le<T, U>(x: T, y: T) -> U;

    /// Tests if `x` is greater than `y`, elementwise.
    ///
    /// Returns `0` for false and `!0` for true.
    ///
    /// `T` must be a vector of floating-point primitive types.
    ///
    /// `U` must be a vector of integers with the same number of elements and element size as `T`.
    pub fn simd_gt<T, U>(x: T, y: T) -> U;

    /// Tests if `x` is greater than or equal to `y`, elementwise.
    ///
    /// Returns `0` for false and `!0` for true.
    ///
    /// `T` must be a vector of floating-point primitive types.
    ///
    /// `U` must be a vector of integers with the same number of elements and element size as `T`.
    pub fn simd_ge<T, U>(x: T, y: T) -> U;

    /// Shuffle two vectors by const indices.
    ///
    /// Concatenates `x` and `y`, then returns a new vector such that each element is selected from
    /// the concatenation by the matching index in `idx`.
    ///
    /// `T` must be a vector.
    ///
    /// `U` must be a const array of `i32`s.
    ///
    /// `V` must be a vector with the same element type as `T` and the same length as `U`.
    pub fn simd_shuffle<T, U, V>(x: T, y: T, idx: U) -> V;

    /// Read a vector of pointers.
    ///
    /// For each pointer in `ptr`, if the corresponding value in `mask` is `!0`, read the pointer.
    /// Otherwise if the corresponding value in `mask` is `0`, return the corresponding value from
    /// `val`.
    ///
    /// `T` must be a vector.
    ///
    /// `U` must be a vector of pointers to the element type of `T`, with the same length as `T`.
    ///
    /// `V` must be a vector of integers with the same length as `T` (but any element size).
    ///
    /// # Safety
    /// Unmasked values in `T` must be readable as if by `<ptr>::read` (e.g. aligned to the element
    /// type).
    ///
    /// `mask` must only contain `0` or `!0` values.
    pub fn simd_gather<T, U, V>(val: T, ptr: U, mask: V) -> T;

    /// Write to a vector of pointers.
    ///
    /// For each pointer in `ptr`, if the corresponding value in `mask` is `!0`, write the
    /// corresponding value in `val` to the pointer.
    /// Otherwise if the corresponding value in `mask` is `0`, do nothing.
    ///
    /// `T` must be a vector.
    ///
    /// `U` must be a vector of pointers to the element type of `T`, with the same length as `T`.
    ///
    /// `V` must be a vector of integers with the same length as `T` (but any element size).
    ///
    /// # Safety
    /// Unmasked values in `T` must be writeable as if by `<ptr>::write` (e.g. aligned to the element
    /// type).
    ///
    /// `mask` must only contain `0` or `!0` values.
    pub fn simd_scatter<T, U, V>(val: T, ptr: U, mask: V);

    /// Add two simd vectors elementwise, with saturation.
    ///
    /// `T` must be a vector of integer primitive types.
    pub fn simd_saturating_add<T>(x: T, y: T) -> T;

    /// Subtract two simd vectors elementwise, with saturation.
    ///
    /// Subtract `rhs` from `lhs`.
    ///
    /// `T` must be a vector of integer primitive types.
    pub fn simd_saturating_sub<T>(lhs: T, rhs: T) -> T;

    /// Add elements within a vector from left to right.
    ///
    /// Starting with the value `y`, add the elements of `x` and accumulate.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    pub fn simd_reduce_add_ordered<T, U>(x: T, y: U) -> U;

    /// Multiply elements within a vector from left to right.
    ///
    /// Starting with the value `y`, multiply the elements of `x` and accumulate.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    pub fn simd_reduce_mul_ordered<T, U>(x: T, y: U) -> U;

    /// Check if all mask values are true.
    ///
    /// `T` must be a vector of integer primitive types.
    ///
    /// # Safety
    /// `x` must contain only `0` or `!0`.
    pub fn simd_reduce_all<T>(x: T) -> bool;

    /// Check if all mask values are true.
    ///
    /// `T` must be a vector of integer primitive types.
    ///
    /// # Safety
    /// `x` must contain only `0` or `!0`.
    pub fn simd_reduce_any<T>(x: T) -> bool;

    /// Return the maximum element of a vector.
    ///
    /// For floating-point values, uses IEEE-754 `maxNum`.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    pub fn simd_reduce_max<T, U>(x: T) -> U;

    /// Return the minimum element of a vector.
    ///
    /// For floating-point values, uses IEEE-754 `minNum`.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    pub fn simd_reduce_min<T, U>(x: T) -> U;

    /// Logical "and" all elements together.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    pub fn simd_reduce_and<T, U>(x: T) -> U;

    /// Logical "or" all elements together.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    pub fn simd_reduce_or<T, U>(x: T) -> U;

    /// Logical "exclusive or" all elements together.
    ///
    /// `T` must be a vector of integer or floating-point primitive types.
    ///
    /// `U` must be the element type of `T`.
    pub fn simd_reduce_xor<T, U>(x: T) -> U;

    /// Truncate an integer vector to a bitmask.
    ///
    /// Each element is truncated to a single bit and packed into the result.
    ///
    /// The bit order depends on the byte endianness.
    /// The bitmask is always packed into the smallest/first bits, but the order is LSB-first for
    /// little endian and MSB-first for big endian.
    /// In other words, the LSB corresponds to the first vector element for little endian,
    /// and the last vector element for big endian.
    ///
    /// `T` must be an integer vector.
    ///
    /// `U` must be either the smallest unsigned integer with at least as many bits as the length
    /// of `T`, or the smallest array of `u8` with as many bits as the length of `T`.
    ///
    /// # Safety
    /// `x` must contain only `0` and `!0`.
    pub fn simd_bitmask<T, U>(x: T) -> U;

    /// Select elements from a mask.
    ///
    /// For each element, if the corresponding value in `mask` is `!0`, select the element from
    /// `if_true`.  If the corresponding value in `mask` is `0`, select the element from
    /// `if_false`.
    ///
    /// `M` must be an integer vector.
    ///
    /// `T` must be a vector with the same number of elements as `M`.
    ///
    /// # Safety
    /// `mask` must only contain `0` and `!0`.
    pub fn simd_select<M, T>(mask: M, if_true: T, if_false: T) -> T;

    /// Select elements from a bitmask.
    ///
    /// For each element, if the bit in `mask` is `1`, select the element from
    /// `if_true`.  If the corresponding bit in `mask` is `0`, select the element from
    /// `if_false`.
    ///
    /// The bitmask bit order matches `simd_bitmask`.
    ///
    /// `M` must be an unsigned integer of type matching `simd_bitmask`.
    ///
    /// `T` must be a vector.
    ///
    /// # Safety
    /// `mask` must only contain `0` and `!0`.
    pub fn simd_select_bitmask<M, T>(m: M, yes: T, no: T) -> T;

    /// Elementwise calculates the offset from a pointer vector, potentially wrapping.
    ///
    /// Operates as if by `<ptr>::wrapping_offset`.
    ///
    /// `T` must be a vector of pointers.
    ///
    /// `U` must be a vector of `isize` or `usize` with the same number of elements as `T`.
    pub fn simd_arith_offset<T, U>(ptr: T, offset: U) -> T;

    /// Cast a vector of pointers.
    ///
    /// `T` and `U` must be vectors of pointers with the same number of elements.
    pub fn simd_cast_ptr<T, U>(ptr: T) -> U;

    /// Expose a vector of pointers as a vector of addresses.
    ///
    /// `T` must be a vector of pointers.
    ///
    /// `U` must be a vector of `usize` with the same length as `T`.
    pub fn simd_expose_addr<T, U>(ptr: T) -> U;

    /// Create a vector of pointers from a vector of addresses.
    ///
    /// `T` must be a vector of `usize`.
    ///
    /// `U` must be a vector of pointers, with the same length as `T`.
    pub fn simd_from_exposed_addr<T, U>(addr: T) -> U;

    /// Swap bytes of each element.
    ///
    /// `T` must be a vector of integers.
    pub fn simd_bswap<T>(x: T) -> T;

    /// Reverse bits of each element.
    ///
    /// `T` must be a vector of integers.
    pub fn simd_bitreverse<T>(x: T) -> T;

    /// Count the leading zeros of each element.
    ///
    /// `T` must be a vector of integers.
    pub fn simd_ctlz<T>(x: T) -> T;

    /// Count the trailing zeros of each element.
    ///
    /// `T` must be a vector of integers.
    pub fn simd_cttz<T>(x: T) -> T;
}
