#[link(name="cci_impl_lib", vers="0.0")];

trait uint_helpers {
    fn to(v: uint, f: fn(uint));
}

impl uint: uint_helpers {
    #[inline]
    fn to(v: uint, f: fn(uint)) {
        let mut i = self;
        while i < v {
            f(i);
            i += 1u;
        }
    }
}
