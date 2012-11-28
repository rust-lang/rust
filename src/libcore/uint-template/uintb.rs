pub use inst::{
    div_ceil, div_round, div_floor, iterate,
    next_power_of_two
};

mod inst {
    pub type T = uint;

    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    pub const bits: uint = 32;

    #[cfg(target_arch = "x86_64")]
    pub const bits: uint = 64;

    /**
    * Divide two numbers, return the result, rounded up.
    *
    * # Arguments
    *
    * * x - an integer
    * * y - an integer distinct from 0u
    *
    * # Return value
    *
    * The smallest integer `q` such that `x/y <= q`.
    */
    pub pure fn div_ceil(x: uint, y: uint) -> uint {
        let div = x / y;
        if x % y == 0u { div }
        else { div + 1u }
    }

    /**
    * Divide two numbers, return the result, rounded to the closest integer.
    *
    * # Arguments
    *
    * * x - an integer
    * * y - an integer distinct from 0u
    *
    * # Return value
    *
    * The integer `q` closest to `x/y`.
    */
    pub pure fn div_round(x: uint, y: uint) -> uint {
        let div = x / y;
        if x % y * 2u  < y { div }
        else { div + 1u }
    }

    /**
    * Divide two numbers, return the result, rounded down.
    *
    * Note: This is the same function as `div`.
    *
    * # Arguments
    *
    * * x - an integer
    * * y - an integer distinct from 0u
    *
    * # Return value
    *
    * The smallest integer `q` such that `x/y <= q`. This
    * is either `x/y` or `x/y + 1`.
    */
    pub pure fn div_floor(x: uint, y: uint) -> uint { return x / y; }

    /**
    * Iterate over the range [`lo`..`hi`), or stop when requested
    *
    * # Arguments
    *
    * * lo - The integer at which to start the loop (included)
    * * hi - The integer at which to stop the loop (excluded)
    * * it - A block to execute with each consecutive integer of the range.
    *        Return `true` to continue, `false` to stop.
    *
    * # Return value
    *
    * `true` If execution proceeded correctly, `false` if it was interrupted,
    * that is if `it` returned `false` at any point.
    */
    pub pure fn iterate(lo: uint, hi: uint, it: fn(uint) -> bool) -> bool {
        let mut i = lo;
        while i < hi {
            if (!it(i)) { return false; }
            i += 1u;
        }
        return true;
    }

    /// Returns the smallest power of 2 greater than or equal to `n`
    #[inline(always)]
    pub fn next_power_of_two(n: uint) -> uint {
        let halfbits: uint = sys::size_of::<uint>() * 4u;
        let mut tmp: uint = n - 1u;
        let mut shift: uint = 1u;
        while shift <= halfbits { tmp |= tmp >> shift; shift <<= 1u; }
        return tmp + 1u;
    }

    #[test]
    fn test_next_power_of_two() {
        assert (uint::next_power_of_two(0u) == 0u);
        assert (uint::next_power_of_two(1u) == 1u);
        assert (uint::next_power_of_two(2u) == 2u);
        assert (uint::next_power_of_two(3u) == 4u);
        assert (uint::next_power_of_two(4u) == 4u);
        assert (uint::next_power_of_two(5u) == 8u);
        assert (uint::next_power_of_two(6u) == 8u);
        assert (uint::next_power_of_two(7u) == 8u);
        assert (uint::next_power_of_two(8u) == 8u);
        assert (uint::next_power_of_two(9u) == 16u);
        assert (uint::next_power_of_two(10u) == 16u);
        assert (uint::next_power_of_two(11u) == 16u);
        assert (uint::next_power_of_two(12u) == 16u);
        assert (uint::next_power_of_two(13u) == 16u);
        assert (uint::next_power_of_two(14u) == 16u);
        assert (uint::next_power_of_two(15u) == 16u);
        assert (uint::next_power_of_two(16u) == 16u);
        assert (uint::next_power_of_two(17u) == 32u);
        assert (uint::next_power_of_two(18u) == 32u);
        assert (uint::next_power_of_two(19u) == 32u);
        assert (uint::next_power_of_two(20u) == 32u);
        assert (uint::next_power_of_two(21u) == 32u);
        assert (uint::next_power_of_two(22u) == 32u);
        assert (uint::next_power_of_two(23u) == 32u);
        assert (uint::next_power_of_two(24u) == 32u);
        assert (uint::next_power_of_two(25u) == 32u);
        assert (uint::next_power_of_two(26u) == 32u);
        assert (uint::next_power_of_two(27u) == 32u);
        assert (uint::next_power_of_two(28u) == 32u);
        assert (uint::next_power_of_two(29u) == 32u);
        assert (uint::next_power_of_two(30u) == 32u);
        assert (uint::next_power_of_two(31u) == 32u);
        assert (uint::next_power_of_two(32u) == 32u);
        assert (uint::next_power_of_two(33u) == 64u);
        assert (uint::next_power_of_two(34u) == 64u);
        assert (uint::next_power_of_two(35u) == 64u);
        assert (uint::next_power_of_two(36u) == 64u);
        assert (uint::next_power_of_two(37u) == 64u);
        assert (uint::next_power_of_two(38u) == 64u);
        assert (uint::next_power_of_two(39u) == 64u);
    }

    #[test]
    fn test_overflows() {
        assert (uint::max_value > 0u);
        assert (uint::min_value <= 0u);
        assert (uint::min_value + uint::max_value + 1u == 0u);
    }

    #[test]
    fn test_div() {
        assert(uint::div_floor(3u, 4u) == 0u);
        assert(uint::div_ceil(3u, 4u)  == 1u);
        assert(uint::div_round(3u, 4u) == 1u);
    }
}