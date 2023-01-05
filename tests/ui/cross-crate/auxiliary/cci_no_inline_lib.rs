#![crate_name="cci_no_inline_lib"]


// same as cci_iter_lib, more-or-less, but not marked inline
pub fn iter<F>(v: Vec<usize> , mut f: F) where F: FnMut(usize) {
    let mut i = 0;
    let n = v.len();
    while i < n {
        f(v[i]);
        i += 1;
    }
}
