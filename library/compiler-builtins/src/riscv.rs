intrinsics! {
    // Implementation from gcc
    // https://raw.githubusercontent.com/gcc-mirror/gcc/master/libgcc/config/epiphany/mulsi3.c
    pub extern "C" fn __mulsi3(a: u32, b: u32) -> u32 {
        let (mut a, mut b) = (a, b);
        let mut r = 0;

        while a > 0 {
            if a & 1 > 0 {
                r += b;
            }
            a >>= 1;
            b <<= 1;
        }

        r
    }

    #[cfg(not(target_feature = "m"))]
    pub extern "C" fn __muldi3(a: u64, b: u64) -> u64 {
        let (mut a, mut b) = (a, b);
        let mut r = 0;

        while a > 0 {
            if a & 1 > 0 {
                r += b;
            }
            a >>= 1;
            b <<= 1;
        }

        r
    }
}
