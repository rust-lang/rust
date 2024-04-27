#![crate_name="cci_iter_lib"]

#[inline]
pub fn iter<T, F>(v: &[T], mut f: F) where F: FnMut(&T) {
    let mut i = 0;
    let n = v.len();
    while i < n {
        f(&v[i]);
        i += 1;
    }
}
