#[link(name="cci_no_inline_lib", vers="0.0")];

// same as cci_iter_lib, more-or-less, but not marked inline
fn iter(v: [uint]/~, f: fn(uint)) {
    let mut i = 0u;
    let n = vec::len(v);
    while i < n {
        f(v[i]);
        i += 1u;
    }
}
