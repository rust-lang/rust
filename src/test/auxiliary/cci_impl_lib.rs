#[link(name="cci_impl_lib", vers="0.0")];

trait uint_helpers {
    fn to(v: uint, f: fn(uint));
}

impl helpers of uint_helpers for uint {
    #[inline]
    fn to(v: uint, f: fn(uint)) {
        let mut i = self;
        while i < v {
            f(i);
            i += 1u;
        }
    }
}
