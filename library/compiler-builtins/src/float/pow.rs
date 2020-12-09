use float::Float;

trait Pow: Float {
    /// Returns `a` raised to the power `b`
    fn pow(self, mut b: i32) -> Self {
        let mut a = self;
        let recip = b < 0;
        let mut r = Self::ONE;
        loop {
            if (b & 1) != 0 {
                r *= a;
            }
            b = ((b as u32) >> 1) as i32;
            if b == 0 {
                break;
            }
            a *= a;
        }

        if recip {
            Self::ONE / r
        } else {
            r
        }
    }
}

impl Pow for f32 {}
impl Pow for f64 {}

intrinsics! {
    pub extern "C" fn __powisf2(a: f32, b: i32) -> f32 {
        a.pow(b)
    }

    pub extern "C" fn __powidf2(a: f64, b: i32) -> f64 {
        a.pow(b)
    }
}
