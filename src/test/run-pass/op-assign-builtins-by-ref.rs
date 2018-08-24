fn main() {
    // test compound assignment operators with ref as right-hand side,
    // for each operator, with various types as operands.

    // test AddAssign
    {
        let mut x = 3i8;
        x += &2i8;
        assert_eq!(x, 5i8);
    }

    // test SubAssign
    {
        let mut x = 7i16;
        x -= &4;
        assert_eq!(x, 3i16);
    }

    // test MulAssign
    {
        let mut x = 3f32;
        x *= &3f32;
        assert_eq!(x, 9f32);
    }

    // test DivAssign
    {
        let mut x = 6f64;
        x /= &2f64;
        assert_eq!(x, 3f64);
    }

    // test RemAssign
    {
        let mut x = 7i64;
        x %= &4i64;
        assert_eq!(x, 3i64);
    }

    // test BitOrAssign
    {
        let mut x = 0b1010u8;
        x |= &0b1100u8;
        assert_eq!(x, 0b1110u8);
    }

    // test BitAndAssign
    {
        let mut x = 0b1010u16;
        x &= &0b1100u16;
        assert_eq!(x, 0b1000u16);
    }

    // test BitXorAssign
    {
        let mut x = 0b1010u32;
        x ^= &0b1100u32;
        assert_eq!(x, 0b0110u32);
    }

    // test ShlAssign
    {
        let mut x = 0b1010u64;
        x <<= &2u32;
        assert_eq!(x, 0b101000u64);
    }

    // test ShrAssign
    {
        let mut x = 0b1010u64;
        x >>= &2i16;
        assert_eq!(x, 0b10u64);
    }
}
