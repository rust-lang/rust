impl helpers for uint {
    #[inline]
    fn to(v: uint, f: fn(uint)) {
        let i = self;
        while i < v {
            f(i);
            i += 1u;
        }
    }
}