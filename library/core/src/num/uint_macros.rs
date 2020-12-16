macro_rules! uint_impl {
    ($SelfT:ty, $ActualT:ty, $BITS:expr, $MaxV:expr, $Feature:expr, $EndFeature:expr,
        $rot:expr, $rot_op:expr, $rot_result:expr, $swap_op:expr, $swapped:expr,
        $reversed:expr, $le_bytes:expr, $be_bytes:expr,
        $to_xe_bytes_doc:expr, $from_xe_bytes_doc:expr) => {
        doc_comment! {
            concat!("The smallest value that can be represented by this integer type.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(", stringify!($SelfT), "::MIN, 0);", $EndFeature, "
```"),
            #[stable(feature = "assoc_int_consts", since = "1.43.0")]
            pub const MIN: Self = 0;
        }

        doc_comment! {
            concat!("The largest value that can be represented by this integer type.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(", stringify!($SelfT), "::MAX, ", stringify!($MaxV), ");",
$EndFeature, "
```"),
            #[stable(feature = "assoc_int_consts", since = "1.43.0")]
            pub const MAX: Self = !0;
        }

        doc_comment! {
            concat!("The size of this integer type in bits.

# Examples

```
", $Feature, "#![feature(int_bits_const)]
assert_eq!(", stringify!($SelfT), "::BITS, ", stringify!($BITS), ");",
$EndFeature, "
```"),
            #[unstable(feature = "int_bits_const", issue = "76904")]
            pub const BITS: u32 = $BITS;
        }

        doc_comment! {
            concat!("Converts a string slice in a given base to an integer.

The string is expected to be an optional `+` sign
followed by digits.
Leading and trailing whitespace represent an error.
Digits are a subset of these characters, depending on `radix`:

* `0-9`
* `a-z`
* `A-Z`

# Panics

This function panics if `radix` is not in the range from 2 to 36.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(", stringify!($SelfT), "::from_str_radix(\"A\", 16), Ok(10));",
$EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            pub fn from_str_radix(src: &str, radix: u32) -> Result<Self, ParseIntError> {
                from_str_radix(src, radix)
            }
        }

        doc_comment! {
            concat!("Returns the number of ones in the binary representation of `self`.

# Examples

Basic usage:

```
", $Feature, "let n = 0b01001100", stringify!($SelfT), ";

assert_eq!(n.count_ones(), 3);", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
            #[doc(alias = "popcount")]
            #[doc(alias = "popcnt")]
            #[inline]
            pub const fn count_ones(self) -> u32 {
                intrinsics::ctpop(self as $ActualT) as u32
            }
        }

        doc_comment! {
            concat!("Returns the number of zeros in the binary representation of `self`.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(", stringify!($SelfT), "::MAX.count_zeros(), 0);", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
            #[inline]
            pub const fn count_zeros(self) -> u32 {
                (!self).count_ones()
            }
        }

        doc_comment! {
            concat!("Returns the number of leading zeros in the binary representation of `self`.

# Examples

Basic usage:

```
", $Feature, "let n = ", stringify!($SelfT), "::MAX >> 2;

assert_eq!(n.leading_zeros(), 2);", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
            #[inline]
            pub const fn leading_zeros(self) -> u32 {
                intrinsics::ctlz(self as $ActualT) as u32
            }
        }

        doc_comment! {
            concat!("Returns the number of trailing zeros in the binary representation
of `self`.

# Examples

Basic usage:

```
", $Feature, "let n = 0b0101000", stringify!($SelfT), ";

assert_eq!(n.trailing_zeros(), 3);", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
            #[inline]
            pub const fn trailing_zeros(self) -> u32 {
                intrinsics::cttz(self) as u32
            }
        }

        doc_comment! {
            concat!("Returns the number of leading ones in the binary representation of `self`.

# Examples

Basic usage:

```
", $Feature, "let n = !(", stringify!($SelfT), "::MAX >> 2);

assert_eq!(n.leading_ones(), 2);", $EndFeature, "
```"),
            #[stable(feature = "leading_trailing_ones", since = "1.46.0")]
            #[rustc_const_stable(feature = "leading_trailing_ones", since = "1.46.0")]
            #[inline]
            pub const fn leading_ones(self) -> u32 {
                (!self).leading_zeros()
            }
        }

        doc_comment! {
            concat!("Returns the number of trailing ones in the binary representation
of `self`.

# Examples

Basic usage:

```
", $Feature, "let n = 0b1010111", stringify!($SelfT), ";

assert_eq!(n.trailing_ones(), 3);", $EndFeature, "
```"),
            #[stable(feature = "leading_trailing_ones", since = "1.46.0")]
            #[rustc_const_stable(feature = "leading_trailing_ones", since = "1.46.0")]
            #[inline]
            pub const fn trailing_ones(self) -> u32 {
                (!self).trailing_zeros()
            }
        }

        doc_comment! {
            concat!("Shifts the bits to the left by a specified amount, `n`,
wrapping the truncated bits to the end of the resulting integer.

Please note this isn't the same operation as the `<<` shifting operator!

# Examples

Basic usage:

```
let n = ", $rot_op, stringify!($SelfT), ";
let m = ", $rot_result, ";

assert_eq!(n.rotate_left(", $rot, "), m);
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn rotate_left(self, n: u32) -> Self {
                intrinsics::rotate_left(self, n as $SelfT)
            }
        }

        doc_comment! {
            concat!("Shifts the bits to the right by a specified amount, `n`,
wrapping the truncated bits to the beginning of the resulting
integer.

Please note this isn't the same operation as the `>>` shifting operator!

# Examples

Basic usage:

```
let n = ", $rot_result, stringify!($SelfT), ";
let m = ", $rot_op, ";

assert_eq!(n.rotate_right(", $rot, "), m);
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn rotate_right(self, n: u32) -> Self {
                intrinsics::rotate_right(self, n as $SelfT)
            }
        }

        doc_comment! {
            concat!("
Reverses the byte order of the integer.

# Examples

Basic usage:

```
let n = ", $swap_op, stringify!($SelfT), ";
let m = n.swap_bytes();

assert_eq!(m, ", $swapped, ");
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
            #[inline]
            pub const fn swap_bytes(self) -> Self {
                intrinsics::bswap(self as $ActualT) as Self
            }
        }

        doc_comment! {
            concat!("Reverses the order of bits in the integer. The least significant bit becomes the most significant bit,
                second least-significant bit becomes second most-significant bit, etc.

# Examples

Basic usage:

```
let n = ", $swap_op, stringify!($SelfT), ";
let m = n.reverse_bits();

assert_eq!(m, ", $reversed, ");
assert_eq!(0, 0", stringify!($SelfT), ".reverse_bits());
```"),
            #[stable(feature = "reverse_bits", since = "1.37.0")]
            #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
            #[inline]
            #[must_use]
            pub const fn reverse_bits(self) -> Self {
                intrinsics::bitreverse(self as $ActualT) as Self
            }
        }

        doc_comment! {
            concat!("Converts an integer from big endian to the target's endianness.

On big endian this is a no-op. On little endian the bytes are
swapped.

# Examples

Basic usage:

```
", $Feature, "let n = 0x1A", stringify!($SelfT), ";

if cfg!(target_endian = \"big\") {
    assert_eq!(", stringify!($SelfT), "::from_be(n), n)
} else {
    assert_eq!(", stringify!($SelfT), "::from_be(n), n.swap_bytes())
}", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
            #[inline]
            pub const fn from_be(x: Self) -> Self {
                #[cfg(target_endian = "big")]
                {
                    x
                }
                #[cfg(not(target_endian = "big"))]
                {
                    x.swap_bytes()
                }
            }
        }

        doc_comment! {
            concat!("Converts an integer from little endian to the target's endianness.

On little endian this is a no-op. On big endian the bytes are
swapped.

# Examples

Basic usage:

```
", $Feature, "let n = 0x1A", stringify!($SelfT), ";

if cfg!(target_endian = \"little\") {
    assert_eq!(", stringify!($SelfT), "::from_le(n), n)
} else {
    assert_eq!(", stringify!($SelfT), "::from_le(n), n.swap_bytes())
}", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
            #[inline]
            pub const fn from_le(x: Self) -> Self {
                #[cfg(target_endian = "little")]
                {
                    x
                }
                #[cfg(not(target_endian = "little"))]
                {
                    x.swap_bytes()
                }
            }
        }

        doc_comment! {
            concat!("Converts `self` to big endian from the target's endianness.

On big endian this is a no-op. On little endian the bytes are
swapped.

# Examples

Basic usage:

```
", $Feature, "let n = 0x1A", stringify!($SelfT), ";

if cfg!(target_endian = \"big\") {
    assert_eq!(n.to_be(), n)
} else {
    assert_eq!(n.to_be(), n.swap_bytes())
}", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
            #[inline]
            pub const fn to_be(self) -> Self { // or not to be?
                #[cfg(target_endian = "big")]
                {
                    self
                }
                #[cfg(not(target_endian = "big"))]
                {
                    self.swap_bytes()
                }
            }
        }

        doc_comment! {
            concat!("Converts `self` to little endian from the target's endianness.

On little endian this is a no-op. On big endian the bytes are
swapped.

# Examples

Basic usage:

```
", $Feature, "let n = 0x1A", stringify!($SelfT), ";

if cfg!(target_endian = \"little\") {
    assert_eq!(n.to_le(), n)
} else {
    assert_eq!(n.to_le(), n.swap_bytes())
}", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_math", since = "1.32.0")]
            #[inline]
            pub const fn to_le(self) -> Self {
                #[cfg(target_endian = "little")]
                {
                    self
                }
                #[cfg(not(target_endian = "little"))]
                {
                    self.swap_bytes()
                }
            }
        }

        doc_comment! {
            concat!("Checked integer addition. Computes `self + rhs`, returning `None`
if overflow occurred.

# Examples

Basic usage:

```
", $Feature, "assert_eq!((", stringify!($SelfT), "::MAX - 2).checked_add(1), ",
"Some(", stringify!($SelfT), "::MAX - 1));
assert_eq!((", stringify!($SelfT), "::MAX - 2).checked_add(3), None);", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn checked_add(self, rhs: Self) -> Option<Self> {
                let (a, b) = self.overflowing_add(rhs);
                if unlikely!(b) {None} else {Some(a)}
            }
        }

        doc_comment! {
            concat!("Unchecked integer addition. Computes `self + rhs`, assuming overflow
cannot occur. This results in undefined behavior when `self + rhs > ", stringify!($SelfT),
"::MAX` or `self + rhs < ", stringify!($SelfT), "::MIN`."),
            #[unstable(
                feature = "unchecked_math",
                reason = "niche optimization path",
                issue = "none",
            )]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub unsafe fn unchecked_add(self, rhs: Self) -> Self {
                // SAFETY: the caller must uphold the safety contract for
                // `unchecked_add`.
                unsafe { intrinsics::unchecked_add(self, rhs) }
            }
        }

        doc_comment! {
            concat!("Checked integer subtraction. Computes `self - rhs`, returning
`None` if overflow occurred.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(1", stringify!($SelfT), ".checked_sub(1), Some(0));
assert_eq!(0", stringify!($SelfT), ".checked_sub(1), None);", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
                let (a, b) = self.overflowing_sub(rhs);
                if unlikely!(b) {None} else {Some(a)}
            }
        }

        doc_comment! {
            concat!("Unchecked integer subtraction. Computes `self - rhs`, assuming overflow
cannot occur. This results in undefined behavior when `self - rhs > ", stringify!($SelfT),
"::MAX` or `self - rhs < ", stringify!($SelfT), "::MIN`."),
            #[unstable(
                feature = "unchecked_math",
                reason = "niche optimization path",
                issue = "none",
            )]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub unsafe fn unchecked_sub(self, rhs: Self) -> Self {
                // SAFETY: the caller must uphold the safety contract for
                // `unchecked_sub`.
                unsafe { intrinsics::unchecked_sub(self, rhs) }
            }
        }

        doc_comment! {
            concat!("Checked integer multiplication. Computes `self * rhs`, returning
`None` if overflow occurred.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(5", stringify!($SelfT), ".checked_mul(1), Some(5));
assert_eq!(", stringify!($SelfT), "::MAX.checked_mul(2), None);", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
                let (a, b) = self.overflowing_mul(rhs);
                if unlikely!(b) {None} else {Some(a)}
            }
        }

        doc_comment! {
            concat!("Unchecked integer multiplication. Computes `self * rhs`, assuming overflow
cannot occur. This results in undefined behavior when `self * rhs > ", stringify!($SelfT),
"::MAX` or `self * rhs < ", stringify!($SelfT), "::MIN`."),
            #[unstable(
                feature = "unchecked_math",
                reason = "niche optimization path",
                issue = "none",
            )]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub unsafe fn unchecked_mul(self, rhs: Self) -> Self {
                // SAFETY: the caller must uphold the safety contract for
                // `unchecked_mul`.
                unsafe { intrinsics::unchecked_mul(self, rhs) }
            }
        }

        doc_comment! {
            concat!("Checked integer division. Computes `self / rhs`, returning `None`
if `rhs == 0`.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(128", stringify!($SelfT), ".checked_div(2), Some(64));
assert_eq!(1", stringify!($SelfT), ".checked_div(0), None);", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_unstable(feature = "const_checked_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn checked_div(self, rhs: Self) -> Option<Self> {
                if unlikely!(rhs == 0) {
                    None
                } else {
                    // SAFETY: div by zero has been checked above and unsigned types have no other
                    // failure modes for division
                    Some(unsafe { intrinsics::unchecked_div(self, rhs) })
                }
            }
        }

        doc_comment! {
            concat!("Checked Euclidean division. Computes `self.div_euclid(rhs)`, returning `None`
if `rhs == 0`.

# Examples

Basic usage:

```
assert_eq!(128", stringify!($SelfT), ".checked_div_euclid(2), Some(64));
assert_eq!(1", stringify!($SelfT), ".checked_div_euclid(0), None);
```"),
            #[stable(feature = "euclidean_division", since = "1.38.0")]
            #[rustc_const_unstable(feature = "const_euclidean_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn checked_div_euclid(self, rhs: Self) -> Option<Self> {
                if unlikely!(rhs == 0) {
                    None
                } else {
                    Some(self.div_euclid(rhs))
                }
            }
        }


        doc_comment! {
            concat!("Checked integer remainder. Computes `self % rhs`, returning `None`
if `rhs == 0`.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(5", stringify!($SelfT), ".checked_rem(2), Some(1));
assert_eq!(5", stringify!($SelfT), ".checked_rem(0), None);", $EndFeature, "
```"),
            #[stable(feature = "wrapping", since = "1.7.0")]
            #[rustc_const_unstable(feature = "const_checked_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn checked_rem(self, rhs: Self) -> Option<Self> {
                if unlikely!(rhs == 0) {
                    None
                } else {
                    // SAFETY: div by zero has been checked above and unsigned types have no other
                    // failure modes for division
                    Some(unsafe { intrinsics::unchecked_rem(self, rhs) })
                }
            }
        }

        doc_comment! {
            concat!("Checked Euclidean modulo. Computes `self.rem_euclid(rhs)`, returning `None`
if `rhs == 0`.

# Examples

Basic usage:

```
assert_eq!(5", stringify!($SelfT), ".checked_rem_euclid(2), Some(1));
assert_eq!(5", stringify!($SelfT), ".checked_rem_euclid(0), None);
```"),
            #[stable(feature = "euclidean_division", since = "1.38.0")]
            #[rustc_const_unstable(feature = "const_euclidean_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn checked_rem_euclid(self, rhs: Self) -> Option<Self> {
                if unlikely!(rhs == 0) {
                    None
                } else {
                    Some(self.rem_euclid(rhs))
                }
            }
        }

        doc_comment! {
            concat!("Checked negation. Computes `-self`, returning `None` unless `self ==
0`.

Note that negating any positive integer will overflow.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(0", stringify!($SelfT), ".checked_neg(), Some(0));
assert_eq!(1", stringify!($SelfT), ".checked_neg(), None);", $EndFeature, "
```"),
            #[stable(feature = "wrapping", since = "1.7.0")]
            #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
            #[inline]
            pub const fn checked_neg(self) -> Option<Self> {
                let (a, b) = self.overflowing_neg();
                if unlikely!(b) {None} else {Some(a)}
            }
        }

        doc_comment! {
            concat!("Checked shift left. Computes `self << rhs`, returning `None`
if `rhs` is larger than or equal to the number of bits in `self`.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(0x1", stringify!($SelfT), ".checked_shl(4), Some(0x10));
assert_eq!(0x10", stringify!($SelfT), ".checked_shl(129), None);", $EndFeature, "
```"),
            #[stable(feature = "wrapping", since = "1.7.0")]
            #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn checked_shl(self, rhs: u32) -> Option<Self> {
                let (a, b) = self.overflowing_shl(rhs);
                if unlikely!(b) {None} else {Some(a)}
            }
        }

        doc_comment! {
            concat!("Checked shift right. Computes `self >> rhs`, returning `None`
if `rhs` is larger than or equal to the number of bits in `self`.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(0x10", stringify!($SelfT), ".checked_shr(4), Some(0x1));
assert_eq!(0x10", stringify!($SelfT), ".checked_shr(129), None);", $EndFeature, "
```"),
            #[stable(feature = "wrapping", since = "1.7.0")]
            #[rustc_const_stable(feature = "const_checked_int_methods", since = "1.47.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn checked_shr(self, rhs: u32) -> Option<Self> {
                let (a, b) = self.overflowing_shr(rhs);
                if unlikely!(b) {None} else {Some(a)}
            }
        }

        doc_comment! {
            concat!("Checked exponentiation. Computes `self.pow(exp)`, returning `None` if
overflow occurred.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(2", stringify!($SelfT), ".checked_pow(5), Some(32));
assert_eq!(", stringify!($SelfT), "::MAX.checked_pow(2), None);", $EndFeature, "
```"),
            #[stable(feature = "no_panic_pow", since = "1.34.0")]
            #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn checked_pow(self, mut exp: u32) -> Option<Self> {
                if exp == 0 {
                    return Some(1);
                }
                let mut base = self;
                let mut acc: Self = 1;

                while exp > 1 {
                    if (exp & 1) == 1 {
                        acc = try_opt!(acc.checked_mul(base));
                    }
                    exp /= 2;
                    base = try_opt!(base.checked_mul(base));
                }

                // since exp!=0, finally the exp must be 1.
                // Deal with the final bit of the exponent separately, since
                // squaring the base afterwards is not necessary and may cause a
                // needless overflow.

                Some(try_opt!(acc.checked_mul(base)))
            }
        }

        doc_comment! {
            concat!("Saturating integer addition. Computes `self + rhs`, saturating at
the numeric bounds instead of overflowing.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(100", stringify!($SelfT), ".saturating_add(1), 101);
assert_eq!(", stringify!($SelfT), "::MAX.saturating_add(127), ", stringify!($SelfT), "::MAX);",
$EndFeature, "
```"),

            #[stable(feature = "rust1", since = "1.0.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[rustc_const_stable(feature = "const_saturating_int_methods", since = "1.47.0")]
            #[inline]
            pub const fn saturating_add(self, rhs: Self) -> Self {
                intrinsics::saturating_add(self, rhs)
            }
        }

        doc_comment! {
            concat!("Saturating integer subtraction. Computes `self - rhs`, saturating
at the numeric bounds instead of overflowing.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(100", stringify!($SelfT), ".saturating_sub(27), 73);
assert_eq!(13", stringify!($SelfT), ".saturating_sub(127), 0);", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[rustc_const_stable(feature = "const_saturating_int_methods", since = "1.47.0")]
            #[inline]
            pub const fn saturating_sub(self, rhs: Self) -> Self {
                intrinsics::saturating_sub(self, rhs)
            }
        }

        doc_comment! {
            concat!("Saturating integer multiplication. Computes `self * rhs`,
saturating at the numeric bounds instead of overflowing.

# Examples

Basic usage:

```
", $Feature, "
assert_eq!(2", stringify!($SelfT), ".saturating_mul(10), 20);
assert_eq!((", stringify!($SelfT), "::MAX).saturating_mul(10), ", stringify!($SelfT),
"::MAX);", $EndFeature, "
```"),
            #[stable(feature = "wrapping", since = "1.7.0")]
            #[rustc_const_stable(feature = "const_saturating_int_methods", since = "1.47.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn saturating_mul(self, rhs: Self) -> Self {
                match self.checked_mul(rhs) {
                    Some(x) => x,
                    None => Self::MAX,
                }
            }
        }

        doc_comment! {
            concat!("Saturating integer exponentiation. Computes `self.pow(exp)`,
saturating at the numeric bounds instead of overflowing.

# Examples

Basic usage:

```
", $Feature, "
assert_eq!(4", stringify!($SelfT), ".saturating_pow(3), 64);
assert_eq!(", stringify!($SelfT), "::MAX.saturating_pow(2), ", stringify!($SelfT), "::MAX);",
$EndFeature, "
```"),
            #[stable(feature = "no_panic_pow", since = "1.34.0")]
            #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn saturating_pow(self, exp: u32) -> Self {
                match self.checked_pow(exp) {
                    Some(x) => x,
                    None => Self::MAX,
                }
            }
        }

        doc_comment! {
            concat!("Wrapping (modular) addition. Computes `self + rhs`,
wrapping around at the boundary of the type.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(200", stringify!($SelfT), ".wrapping_add(55), 255);
assert_eq!(200", stringify!($SelfT), ".wrapping_add(", stringify!($SelfT), "::MAX), 199);",
$EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn wrapping_add(self, rhs: Self) -> Self {
                intrinsics::wrapping_add(self, rhs)
            }
        }

        doc_comment! {
            concat!("Wrapping (modular) subtraction. Computes `self - rhs`,
wrapping around at the boundary of the type.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(100", stringify!($SelfT), ".wrapping_sub(100), 0);
assert_eq!(100", stringify!($SelfT), ".wrapping_sub(", stringify!($SelfT), "::MAX), 101);",
$EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn wrapping_sub(self, rhs: Self) -> Self {
                intrinsics::wrapping_sub(self, rhs)
            }
        }

        /// Wrapping (modular) multiplication. Computes `self *
        /// rhs`, wrapping around at the boundary of the type.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// Please note that this example is shared between integer types.
        /// Which explains why `u8` is used here.
        ///
        /// ```
        /// assert_eq!(10u8.wrapping_mul(12), 120);
        /// assert_eq!(25u8.wrapping_mul(12), 44);
        /// ```
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
        #[inline]
        pub const fn wrapping_mul(self, rhs: Self) -> Self {
            intrinsics::wrapping_mul(self, rhs)
        }

        doc_comment! {
            concat!("Wrapping (modular) division. Computes `self / rhs`.
Wrapped division on unsigned types is just normal division.
There's no way wrapping could ever happen.
This function exists, so that all operations
are accounted for in the wrapping operations.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(100", stringify!($SelfT), ".wrapping_div(10), 10);", $EndFeature, "
```"),
            #[stable(feature = "num_wrapping", since = "1.2.0")]
            #[rustc_const_unstable(feature = "const_wrapping_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn wrapping_div(self, rhs: Self) -> Self {
                self / rhs
            }
        }

        doc_comment! {
            concat!("Wrapping Euclidean division. Computes `self.div_euclid(rhs)`.
Wrapped division on unsigned types is just normal division.
There's no way wrapping could ever happen.
This function exists, so that all operations
are accounted for in the wrapping operations.
Since, for the positive integers, all common
definitions of division are equal, this
is exactly equal to `self.wrapping_div(rhs)`.

# Examples

Basic usage:

```
assert_eq!(100", stringify!($SelfT), ".wrapping_div_euclid(10), 10);
```"),
            #[stable(feature = "euclidean_division", since = "1.38.0")]
            #[rustc_const_unstable(feature = "const_euclidean_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn wrapping_div_euclid(self, rhs: Self) -> Self {
                self / rhs
            }
        }

        doc_comment! {
            concat!("Wrapping (modular) remainder. Computes `self % rhs`.
Wrapped remainder calculation on unsigned types is
just the regular remainder calculation.
There's no way wrapping could ever happen.
This function exists, so that all operations
are accounted for in the wrapping operations.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(100", stringify!($SelfT), ".wrapping_rem(10), 0);", $EndFeature, "
```"),
            #[stable(feature = "num_wrapping", since = "1.2.0")]
            #[rustc_const_unstable(feature = "const_wrapping_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn wrapping_rem(self, rhs: Self) -> Self {
                self % rhs
            }
        }

        doc_comment! {
            concat!("Wrapping Euclidean modulo. Computes `self.rem_euclid(rhs)`.
Wrapped modulo calculation on unsigned types is
just the regular remainder calculation.
There's no way wrapping could ever happen.
This function exists, so that all operations
are accounted for in the wrapping operations.
Since, for the positive integers, all common
definitions of division are equal, this
is exactly equal to `self.wrapping_rem(rhs)`.

# Examples

Basic usage:

```
assert_eq!(100", stringify!($SelfT), ".wrapping_rem_euclid(10), 0);
```"),
            #[stable(feature = "euclidean_division", since = "1.38.0")]
            #[rustc_const_unstable(feature = "const_euclidean_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn wrapping_rem_euclid(self, rhs: Self) -> Self {
                self % rhs
            }
        }

        /// Wrapping (modular) negation. Computes `-self`,
        /// wrapping around at the boundary of the type.
        ///
        /// Since unsigned types do not have negative equivalents
        /// all applications of this function will wrap (except for `-0`).
        /// For values smaller than the corresponding signed type's maximum
        /// the result is the same as casting the corresponding signed value.
        /// Any larger values are equivalent to `MAX + 1 - (val - MAX - 1)` where
        /// `MAX` is the corresponding signed type's maximum.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// Please note that this example is shared between integer types.
        /// Which explains why `i8` is used here.
        ///
        /// ```
        /// assert_eq!(100i8.wrapping_neg(), -100);
        /// assert_eq!((-128i8).wrapping_neg(), -128);
        /// ```
        #[stable(feature = "num_wrapping", since = "1.2.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
        #[inline]
        pub const fn wrapping_neg(self) -> Self {
            self.overflowing_neg().0
        }

        doc_comment! {
            concat!("Panic-free bitwise shift-left; yields `self << mask(rhs)`,
where `mask` removes any high-order bits of `rhs` that
would cause the shift to exceed the bitwidth of the type.

Note that this is *not* the same as a rotate-left; the
RHS of a wrapping shift-left is restricted to the range
of the type, rather than the bits shifted out of the LHS
being returned to the other end. The primitive integer
types all implement a [`rotate_left`](#method.rotate_left) function,
which may be what you want instead.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(1", stringify!($SelfT), ".wrapping_shl(7), 128);
assert_eq!(1", stringify!($SelfT), ".wrapping_shl(128), 1);", $EndFeature, "
```"),
            #[stable(feature = "num_wrapping", since = "1.2.0")]
            #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn wrapping_shl(self, rhs: u32) -> Self {
                // SAFETY: the masking by the bitsize of the type ensures that we do not shift
                // out of bounds
                unsafe {
                    intrinsics::unchecked_shl(self, (rhs & ($BITS - 1)) as $SelfT)
                }
            }
        }

        doc_comment! {
            concat!("Panic-free bitwise shift-right; yields `self >> mask(rhs)`,
where `mask` removes any high-order bits of `rhs` that
would cause the shift to exceed the bitwidth of the type.

Note that this is *not* the same as a rotate-right; the
RHS of a wrapping shift-right is restricted to the range
of the type, rather than the bits shifted out of the LHS
being returned to the other end. The primitive integer
types all implement a [`rotate_right`](#method.rotate_right) function,
which may be what you want instead.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(128", stringify!($SelfT), ".wrapping_shr(7), 1);
assert_eq!(128", stringify!($SelfT), ".wrapping_shr(128), 128);", $EndFeature, "
```"),
            #[stable(feature = "num_wrapping", since = "1.2.0")]
            #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn wrapping_shr(self, rhs: u32) -> Self {
                // SAFETY: the masking by the bitsize of the type ensures that we do not shift
                // out of bounds
                unsafe {
                    intrinsics::unchecked_shr(self, (rhs & ($BITS - 1)) as $SelfT)
                }
            }
        }

        doc_comment! {
            concat!("Wrapping (modular) exponentiation. Computes `self.pow(exp)`,
wrapping around at the boundary of the type.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(3", stringify!($SelfT), ".wrapping_pow(5), 243);
assert_eq!(3u8.wrapping_pow(6), 217);", $EndFeature, "
```"),
            #[stable(feature = "no_panic_pow", since = "1.34.0")]
            #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn wrapping_pow(self, mut exp: u32) -> Self {
                if exp == 0 {
                    return 1;
                }
                let mut base = self;
                let mut acc: Self = 1;

                while exp > 1 {
                    if (exp & 1) == 1 {
                        acc = acc.wrapping_mul(base);
                    }
                    exp /= 2;
                    base = base.wrapping_mul(base);
                }

                // since exp!=0, finally the exp must be 1.
                // Deal with the final bit of the exponent separately, since
                // squaring the base afterwards is not necessary and may cause a
                // needless overflow.
                acc.wrapping_mul(base)
            }
        }

        doc_comment! {
            concat!("Calculates `self` + `rhs`

Returns a tuple of the addition along with a boolean indicating
whether an arithmetic overflow would occur. If an overflow would
have occurred then the wrapped value is returned.

# Examples

Basic usage

```
", $Feature, "
assert_eq!(5", stringify!($SelfT), ".overflowing_add(2), (7, false));
assert_eq!(", stringify!($SelfT), "::MAX.overflowing_add(1), (0, true));", $EndFeature, "
```"),
            #[stable(feature = "wrapping", since = "1.7.0")]
            #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn overflowing_add(self, rhs: Self) -> (Self, bool) {
                let (a, b) = intrinsics::add_with_overflow(self as $ActualT, rhs as $ActualT);
                (a as Self, b)
            }
        }

        doc_comment! {
            concat!("Calculates `self` - `rhs`

Returns a tuple of the subtraction along with a boolean indicating
whether an arithmetic overflow would occur. If an overflow would
have occurred then the wrapped value is returned.

# Examples

Basic usage

```
", $Feature, "
assert_eq!(5", stringify!($SelfT), ".overflowing_sub(2), (3, false));
assert_eq!(0", stringify!($SelfT), ".overflowing_sub(1), (", stringify!($SelfT), "::MAX, true));",
$EndFeature, "
```"),
            #[stable(feature = "wrapping", since = "1.7.0")]
            #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
                let (a, b) = intrinsics::sub_with_overflow(self as $ActualT, rhs as $ActualT);
                (a as Self, b)
            }
        }

        /// Calculates the multiplication of `self` and `rhs`.
        ///
        /// Returns a tuple of the multiplication along with a boolean
        /// indicating whether an arithmetic overflow would occur. If an
        /// overflow would have occurred then the wrapped value is returned.
        ///
        /// # Examples
        ///
        /// Basic usage:
        ///
        /// Please note that this example is shared between integer types.
        /// Which explains why `u32` is used here.
        ///
        /// ```
        /// assert_eq!(5u32.overflowing_mul(2), (10, false));
        /// assert_eq!(1_000_000_000u32.overflowing_mul(10), (1410065408, true));
        /// ```
        #[stable(feature = "wrapping", since = "1.7.0")]
        #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
        #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
        #[inline]
        pub const fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
            let (a, b) = intrinsics::mul_with_overflow(self as $ActualT, rhs as $ActualT);
            (a as Self, b)
        }

        doc_comment! {
            concat!("Calculates the divisor when `self` is divided by `rhs`.

Returns a tuple of the divisor along with a boolean indicating
whether an arithmetic overflow would occur. Note that for unsigned
integers overflow never occurs, so the second value is always
`false`.

# Panics

This function will panic if `rhs` is 0.

# Examples

Basic usage

```
", $Feature, "assert_eq!(5", stringify!($SelfT), ".overflowing_div(2), (2, false));", $EndFeature, "
```"),
            #[inline]
            #[stable(feature = "wrapping", since = "1.7.0")]
            #[rustc_const_unstable(feature = "const_overflowing_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            pub const fn overflowing_div(self, rhs: Self) -> (Self, bool) {
                (self / rhs, false)
            }
        }

        doc_comment! {
            concat!("Calculates the quotient of Euclidean division `self.div_euclid(rhs)`.

Returns a tuple of the divisor along with a boolean indicating
whether an arithmetic overflow would occur. Note that for unsigned
integers overflow never occurs, so the second value is always
`false`.
Since, for the positive integers, all common
definitions of division are equal, this
is exactly equal to `self.overflowing_div(rhs)`.

# Panics

This function will panic if `rhs` is 0.

# Examples

Basic usage

```
assert_eq!(5", stringify!($SelfT), ".overflowing_div_euclid(2), (2, false));
```"),
            #[inline]
            #[stable(feature = "euclidean_division", since = "1.38.0")]
            #[rustc_const_unstable(feature = "const_euclidean_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            pub const fn overflowing_div_euclid(self, rhs: Self) -> (Self, bool) {
                (self / rhs, false)
            }
        }

        doc_comment! {
            concat!("Calculates the remainder when `self` is divided by `rhs`.

Returns a tuple of the remainder after dividing along with a boolean
indicating whether an arithmetic overflow would occur. Note that for
unsigned integers overflow never occurs, so the second value is
always `false`.

# Panics

This function will panic if `rhs` is 0.

# Examples

Basic usage

```
", $Feature, "assert_eq!(5", stringify!($SelfT), ".overflowing_rem(2), (1, false));", $EndFeature, "
```"),
            #[inline]
            #[stable(feature = "wrapping", since = "1.7.0")]
            #[rustc_const_unstable(feature = "const_overflowing_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            pub const fn overflowing_rem(self, rhs: Self) -> (Self, bool) {
                (self % rhs, false)
            }
        }

        doc_comment! {
            concat!("Calculates the remainder `self.rem_euclid(rhs)` as if by Euclidean division.

Returns a tuple of the modulo after dividing along with a boolean
indicating whether an arithmetic overflow would occur. Note that for
unsigned integers overflow never occurs, so the second value is
always `false`.
Since, for the positive integers, all common
definitions of division are equal, this operation
is exactly equal to `self.overflowing_rem(rhs)`.

# Panics

This function will panic if `rhs` is 0.

# Examples

Basic usage

```
assert_eq!(5", stringify!($SelfT), ".overflowing_rem_euclid(2), (1, false));
```"),
            #[inline]
            #[stable(feature = "euclidean_division", since = "1.38.0")]
            #[rustc_const_unstable(feature = "const_euclidean_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            pub const fn overflowing_rem_euclid(self, rhs: Self) -> (Self, bool) {
                (self % rhs, false)
            }
        }

        doc_comment! {
            concat!("Negates self in an overflowing fashion.

Returns `!self + 1` using wrapping operations to return the value
that represents the negation of this unsigned value. Note that for
positive unsigned values overflow always occurs, but negating 0 does
not overflow.

# Examples

Basic usage

```
", $Feature, "assert_eq!(0", stringify!($SelfT), ".overflowing_neg(), (0, false));
assert_eq!(2", stringify!($SelfT), ".overflowing_neg(), (-2i32 as ", stringify!($SelfT),
", true));", $EndFeature, "
```"),
            #[inline]
            #[stable(feature = "wrapping", since = "1.7.0")]
            #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
            pub const fn overflowing_neg(self) -> (Self, bool) {
                ((!self).wrapping_add(1), self != 0)
            }
        }

        doc_comment! {
            concat!("Shifts self left by `rhs` bits.

Returns a tuple of the shifted version of self along with a boolean
indicating whether the shift value was larger than or equal to the
number of bits. If the shift value is too large, then value is
masked (N-1) where N is the number of bits, and this value is then
used to perform the shift.

# Examples

Basic usage

```
", $Feature, "assert_eq!(0x1", stringify!($SelfT), ".overflowing_shl(4), (0x10, false));
assert_eq!(0x1", stringify!($SelfT), ".overflowing_shl(132), (0x10, true));", $EndFeature, "
```"),
            #[stable(feature = "wrapping", since = "1.7.0")]
            #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn overflowing_shl(self, rhs: u32) -> (Self, bool) {
                (self.wrapping_shl(rhs), (rhs > ($BITS - 1)))
            }
        }

        doc_comment! {
            concat!("Shifts self right by `rhs` bits.

Returns a tuple of the shifted version of self along with a boolean
indicating whether the shift value was larger than or equal to the
number of bits. If the shift value is too large, then value is
masked (N-1) where N is the number of bits, and this value is then
used to perform the shift.

# Examples

Basic usage

```
", $Feature, "assert_eq!(0x10", stringify!($SelfT), ".overflowing_shr(4), (0x1, false));
assert_eq!(0x10", stringify!($SelfT), ".overflowing_shr(132), (0x1, true));", $EndFeature, "
```"),
            #[stable(feature = "wrapping", since = "1.7.0")]
            #[rustc_const_stable(feature = "const_wrapping_math", since = "1.32.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn overflowing_shr(self, rhs: u32) -> (Self, bool) {
                (self.wrapping_shr(rhs), (rhs > ($BITS - 1)))
            }
        }

        doc_comment! {
            concat!("Raises self to the power of `exp`, using exponentiation by squaring.

Returns a tuple of the exponentiation along with a bool indicating
whether an overflow happened.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(3", stringify!($SelfT), ".overflowing_pow(5), (243, false));
assert_eq!(3u8.overflowing_pow(6), (217, true));", $EndFeature, "
```"),
            #[stable(feature = "no_panic_pow", since = "1.34.0")]
            #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            pub const fn overflowing_pow(self, mut exp: u32) -> (Self, bool) {
                if exp == 0{
                    return (1,false);
                }
                let mut base = self;
                let mut acc: Self = 1;
                let mut overflown = false;
                // Scratch space for storing results of overflowing_mul.
                let mut r;

                while exp > 1 {
                    if (exp & 1) == 1 {
                        r = acc.overflowing_mul(base);
                        acc = r.0;
                        overflown |= r.1;
                    }
                    exp /= 2;
                    r = base.overflowing_mul(base);
                    base = r.0;
                    overflown |= r.1;
                }

                // since exp!=0, finally the exp must be 1.
                // Deal with the final bit of the exponent separately, since
                // squaring the base afterwards is not necessary and may cause a
                // needless overflow.
                r = acc.overflowing_mul(base);
                r.1 |= overflown;

                r
            }
        }

        doc_comment! {
            concat!("Raises self to the power of `exp`, using exponentiation by squaring.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(2", stringify!($SelfT), ".pow(5), 32);", $EndFeature, "
```"),
        #[stable(feature = "rust1", since = "1.0.0")]
        #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
        #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
        #[inline]
        #[rustc_inherit_overflow_checks]
        pub const fn pow(self, mut exp: u32) -> Self {
            if exp == 0 {
                return 1;
            }
            let mut base = self;
            let mut acc = 1;

            while exp > 1 {
                if (exp & 1) == 1 {
                    acc = acc * base;
                }
                exp /= 2;
                base = base * base;
            }

            // since exp!=0, finally the exp must be 1.
            // Deal with the final bit of the exponent separately, since
            // squaring the base afterwards is not necessary and may cause a
            // needless overflow.
            acc * base
        }
    }

        doc_comment! {
            concat!("Performs Euclidean division.

Since, for the positive integers, all common
definitions of division are equal, this
is exactly equal to `self / rhs`.

# Panics

This function will panic if `rhs` is 0.

# Examples

Basic usage:

```
assert_eq!(7", stringify!($SelfT), ".div_euclid(4), 1); // or any other integer type
```"),
            #[stable(feature = "euclidean_division", since = "1.38.0")]
            #[rustc_const_unstable(feature = "const_euclidean_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            #[rustc_inherit_overflow_checks]
            pub const fn div_euclid(self, rhs: Self) -> Self {
                self / rhs
            }
        }


        doc_comment! {
            concat!("Calculates the least remainder of `self (mod rhs)`.

Since, for the positive integers, all common
definitions of division are equal, this
is exactly equal to `self % rhs`.

# Panics

This function will panic if `rhs` is 0.

# Examples

Basic usage:

```
assert_eq!(7", stringify!($SelfT), ".rem_euclid(4), 3); // or any other integer type
```"),
            #[stable(feature = "euclidean_division", since = "1.38.0")]
            #[rustc_const_unstable(feature = "const_euclidean_int_methods", issue = "53718")]
            #[must_use = "this returns the result of the operation, \
                          without modifying the original"]
            #[inline]
            #[rustc_inherit_overflow_checks]
            pub const fn rem_euclid(self, rhs: Self) -> Self {
                self % rhs
            }
        }

        doc_comment! {
            concat!("Returns `true` if and only if `self == 2^k` for some `k`.

# Examples

Basic usage:

```
", $Feature, "assert!(16", stringify!($SelfT), ".is_power_of_two());
assert!(!10", stringify!($SelfT), ".is_power_of_two());", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_is_power_of_two", since = "1.32.0")]
            #[inline]
            pub const fn is_power_of_two(self) -> bool {
                self.count_ones() == 1
            }
        }

        // Returns one less than next power of two.
        // (For 8u8 next power of two is 8u8 and for 6u8 it is 8u8)
        //
        // 8u8.one_less_than_next_power_of_two() == 7
        // 6u8.one_less_than_next_power_of_two() == 7
        //
        // This method cannot overflow, as in the `next_power_of_two`
        // overflow cases it instead ends up returning the maximum value
        // of the type, and can return 0 for 0.
        #[inline]
        #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
        const fn one_less_than_next_power_of_two(self) -> Self {
            if self <= 1 { return 0; }

            let p = self - 1;
            // SAFETY: Because `p > 0`, it cannot consist entirely of leading zeros.
            // That means the shift is always in-bounds, and some processors
            // (such as intel pre-haswell) have more efficient ctlz
            // intrinsics when the argument is non-zero.
            let z = unsafe { intrinsics::ctlz_nonzero(p) };
            <$SelfT>::MAX >> z
        }

        doc_comment! {
            concat!("Returns the smallest power of two greater than or equal to `self`.

When return value overflows (i.e., `self > (1 << (N-1))` for type
`uN`), it panics in debug mode and return value is wrapped to 0 in
release mode (the only situation in which method can return 0).

# Examples

Basic usage:

```
", $Feature, "assert_eq!(2", stringify!($SelfT), ".next_power_of_two(), 2);
assert_eq!(3", stringify!($SelfT), ".next_power_of_two(), 4);", $EndFeature, "
```"),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
            #[inline]
            #[rustc_inherit_overflow_checks]
            pub const fn next_power_of_two(self) -> Self {
                self.one_less_than_next_power_of_two() + 1
            }
        }

        doc_comment! {
            concat!("Returns the smallest power of two greater than or equal to `n`. If
the next power of two is greater than the type's maximum value,
`None` is returned, otherwise the power of two is wrapped in `Some`.

# Examples

Basic usage:

```
", $Feature, "assert_eq!(2", stringify!($SelfT),
".checked_next_power_of_two(), Some(2));
assert_eq!(3", stringify!($SelfT), ".checked_next_power_of_two(), Some(4));
assert_eq!(", stringify!($SelfT), "::MAX.checked_next_power_of_two(), None);",
$EndFeature, "
```"),
            #[inline]
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
            pub const fn checked_next_power_of_two(self) -> Option<Self> {
                self.one_less_than_next_power_of_two().checked_add(1)
            }
        }

        doc_comment! {
            concat!("Returns the smallest power of two greater than or equal to `n`. If
the next power of two is greater than the type's maximum value,
the return value is wrapped to `0`.

# Examples

Basic usage:

```
#![feature(wrapping_next_power_of_two)]
", $Feature, "
assert_eq!(2", stringify!($SelfT), ".wrapping_next_power_of_two(), 2);
assert_eq!(3", stringify!($SelfT), ".wrapping_next_power_of_two(), 4);
assert_eq!(", stringify!($SelfT), "::MAX.wrapping_next_power_of_two(), 0);",
$EndFeature, "
```"),
            #[unstable(feature = "wrapping_next_power_of_two", issue = "32463",
                       reason = "needs decision on wrapping behaviour")]
            #[rustc_const_stable(feature = "const_int_pow", since = "1.50.0")]
            pub const fn wrapping_next_power_of_two(self) -> Self {
                self.one_less_than_next_power_of_two().wrapping_add(1)
            }
        }

        doc_comment! {
            concat!("Return the memory representation of this integer as a byte array in
big-endian (network) byte order.
",
$to_xe_bytes_doc,
"
# Examples

```
let bytes = ", $swap_op, stringify!($SelfT), ".to_be_bytes();
assert_eq!(bytes, ", $be_bytes, ");
```"),
            #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
            #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
            #[inline]
            pub const fn to_be_bytes(self) -> [u8; mem::size_of::<Self>()] {
                self.to_be().to_ne_bytes()
            }
        }

        doc_comment! {
            concat!("Return the memory representation of this integer as a byte array in
little-endian byte order.
",
$to_xe_bytes_doc,
"
# Examples

```
let bytes = ", $swap_op, stringify!($SelfT), ".to_le_bytes();
assert_eq!(bytes, ", $le_bytes, ");
```"),
            #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
            #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
            #[inline]
            pub const fn to_le_bytes(self) -> [u8; mem::size_of::<Self>()] {
                self.to_le().to_ne_bytes()
            }
        }

        doc_comment! {
            concat!("
Return the memory representation of this integer as a byte array in
native byte order.

As the target platform's native endianness is used, portable code
should use [`to_be_bytes`] or [`to_le_bytes`], as appropriate,
instead.
",
$to_xe_bytes_doc,
"
[`to_be_bytes`]: #method.to_be_bytes
[`to_le_bytes`]: #method.to_le_bytes

# Examples

```
let bytes = ", $swap_op, stringify!($SelfT), ".to_ne_bytes();
assert_eq!(
    bytes,
    if cfg!(target_endian = \"big\") {
        ", $be_bytes, "
    } else {
        ", $le_bytes, "
    }
);
```"),
            #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
            #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
            // SAFETY: const sound because integers are plain old datatypes so we can always
            // transmute them to arrays of bytes
            #[rustc_allow_const_fn_unstable(const_fn_transmute)]
            #[inline]
            pub const fn to_ne_bytes(self) -> [u8; mem::size_of::<Self>()] {
                // SAFETY: integers are plain old datatypes so we can always transmute them to
                // arrays of bytes
                unsafe { mem::transmute(self) }
            }
        }

        doc_comment! {
            concat!("
Return the memory representation of this integer as a byte array in
native byte order.

[`to_ne_bytes`] should be preferred over this whenever possible.

[`to_ne_bytes`]: #method.to_ne_bytes
",

"
# Examples

```
#![feature(num_as_ne_bytes)]
let num = ", $swap_op, stringify!($SelfT), ";
let bytes = num.as_ne_bytes();
assert_eq!(
    bytes,
    if cfg!(target_endian = \"big\") {
        &", $be_bytes, "
    } else {
        &", $le_bytes, "
    }
);
```"),
            #[unstable(feature = "num_as_ne_bytes", issue = "76976")]
            #[inline]
            pub fn as_ne_bytes(&self) -> &[u8; mem::size_of::<Self>()] {
                // SAFETY: integers are plain old datatypes so we can always transmute them to
                // arrays of bytes
                unsafe { &*(self as *const Self as *const _) }
            }
        }

        doc_comment! {
            concat!("Create a native endian integer value from its representation
as a byte array in big endian.
",
$from_xe_bytes_doc,
"
# Examples

```
let value = ", stringify!($SelfT), "::from_be_bytes(", $be_bytes, ");
assert_eq!(value, ", $swap_op, ");
```

When starting from a slice rather than an array, fallible conversion APIs can be used:

```
use std::convert::TryInto;

fn read_be_", stringify!($SelfT), "(input: &mut &[u8]) -> ", stringify!($SelfT), " {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<", stringify!($SelfT), ">());
    *input = rest;
    ", stringify!($SelfT), "::from_be_bytes(int_bytes.try_into().unwrap())
}
```"),
            #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
            #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
            #[inline]
            pub const fn from_be_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
                Self::from_be(Self::from_ne_bytes(bytes))
            }
        }

        doc_comment! {
            concat!("
Create a native endian integer value from its representation
as a byte array in little endian.
",
$from_xe_bytes_doc,
"
# Examples

```
let value = ", stringify!($SelfT), "::from_le_bytes(", $le_bytes, ");
assert_eq!(value, ", $swap_op, ");
```

When starting from a slice rather than an array, fallible conversion APIs can be used:

```
use std::convert::TryInto;

fn read_le_", stringify!($SelfT), "(input: &mut &[u8]) -> ", stringify!($SelfT), " {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<", stringify!($SelfT), ">());
    *input = rest;
    ", stringify!($SelfT), "::from_le_bytes(int_bytes.try_into().unwrap())
}
```"),
            #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
            #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
            #[inline]
            pub const fn from_le_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
                Self::from_le(Self::from_ne_bytes(bytes))
            }
        }

        doc_comment! {
            concat!("Create a native endian integer value from its memory representation
as a byte array in native endianness.

As the target platform's native endianness is used, portable code
likely wants to use [`from_be_bytes`] or [`from_le_bytes`], as
appropriate instead.

[`from_be_bytes`]: #method.from_be_bytes
[`from_le_bytes`]: #method.from_le_bytes
",
$from_xe_bytes_doc,
"
# Examples

```
let value = ", stringify!($SelfT), "::from_ne_bytes(if cfg!(target_endian = \"big\") {
    ", $be_bytes, "
} else {
    ", $le_bytes, "
});
assert_eq!(value, ", $swap_op, ");
```

When starting from a slice rather than an array, fallible conversion APIs can be used:

```
use std::convert::TryInto;

fn read_ne_", stringify!($SelfT), "(input: &mut &[u8]) -> ", stringify!($SelfT), " {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<", stringify!($SelfT), ">());
    *input = rest;
    ", stringify!($SelfT), "::from_ne_bytes(int_bytes.try_into().unwrap())
}
```"),
            #[stable(feature = "int_to_from_bytes", since = "1.32.0")]
            #[rustc_const_stable(feature = "const_int_conversion", since = "1.44.0")]
            // SAFETY: const sound because integers are plain old datatypes so we can always
            // transmute to them
            #[rustc_allow_const_fn_unstable(const_fn_transmute)]
            #[inline]
            pub const fn from_ne_bytes(bytes: [u8; mem::size_of::<Self>()]) -> Self {
                // SAFETY: integers are plain old datatypes so we can always transmute to them
                unsafe { mem::transmute(bytes) }
            }
        }

        doc_comment! {
            concat!("**This method is soft-deprecated.**

Although using it won’t cause compilation warning,
new code should use [`", stringify!($SelfT), "::MIN", "`](#associatedconstant.MIN) instead.

Returns the smallest value that can be represented by this integer type."),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_promotable]
            #[inline(always)]
            #[rustc_const_stable(feature = "const_max_value", since = "1.32.0")]
            pub const fn min_value() -> Self { Self::MIN }
        }

        doc_comment! {
            concat!("**This method is soft-deprecated.**

Although using it won’t cause compilation warning,
new code should use [`", stringify!($SelfT), "::MAX", "`](#associatedconstant.MAX) instead.

Returns the largest value that can be represented by this integer type."),
            #[stable(feature = "rust1", since = "1.0.0")]
            #[rustc_promotable]
            #[inline(always)]
            #[rustc_const_stable(feature = "const_max_value", since = "1.32.0")]
            pub const fn max_value() -> Self { Self::MAX }
        }
    }
}
