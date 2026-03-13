/// Compute a CRC32 checksum using the given polynomial.
///
/// `bit_size` is the number of relevant data bits (8, 16, 32, or 64).
/// Only the low `bit_size` bits of `data` are used; higher bits are ignored.
/// `polynomial` includes the leading 1 bit (e.g. `0x11EDC6F41` for CRC32C).
#[expect(clippy::arithmetic_side_effects)]
pub(crate) fn compute_crc32(crc: u32, data: u64, bit_size: u32, polynomial: u128) -> u32 {
    // Bit-reverse inputs to match hardware CRC conventions.
    let crc = u128::from(crc.reverse_bits());
    // Reverse all 64 bits of `data`, then shift right by `64 - bit_size`. This
    // discards the (now-reversed) higher bits, leaving only the reversed low
    // `bit_size` bits in the lowest positions (with zeros above).
    let v = u128::from(data.reverse_bits() >> (64 - bit_size));

    // Perform polynomial division modulo 2.
    // The algorithm for the division is an adapted version of the
    // schoolbook division algorithm used for normal integer or polynomial
    // division. In this context, the quotient is not calculated, since
    // only the remainder is needed.
    //
    // The algorithm works as follows:
    // 1. Pull down digits until division can be performed. In the context of division
    //    modulo 2 it means locating the most significant digit of the dividend and shifting
    //    the divisor such that the position of the divisors most significand digit and the
    //    dividends most significand digit match.
    // 2. Perform a division and determine the remainder. Since it is arithmetic modulo 2,
    //    this operation is a simple bitwise exclusive or.
    // 3. Repeat steps 1. and 2. until the full remainder is calculated. This is the case
    //    once the degree of the remainder polynomial is smaller than the degree of the
    //    divisor polynomial. In other words, the number of leading zeros of the remainder
    //    is larger than the number of leading zeros of the divisor. It is important to
    //    note that standard arithmetic comparison is not applicable here:
    //    0b10011 / 0b11111 = 0b01100 is a valid division, even though the dividend is
    //    smaller than the divisor.
    let mut dividend = (crc << bit_size) ^ (v << 32);
    while dividend.leading_zeros() <= polynomial.leading_zeros() {
        dividend ^= (polynomial << polynomial.leading_zeros()) >> dividend.leading_zeros();
    }

    u32::try_from(dividend).unwrap().reverse_bits()
}
