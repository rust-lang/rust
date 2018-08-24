#![crate_name="cci_impl_lib"]

pub trait uint_helpers {
    fn to<F>(&self, v: usize, f: F) where F: FnMut(usize);
}

impl uint_helpers for usize {
    #[inline]
    fn to<F>(&self, v: usize, mut f: F) where F: FnMut(usize) {
        let mut i = *self;
        while i < v {
            f(i);
            i += 1;
        }
    }
}
