// same as cci_iter_lib, more-or-less, but not marked inline
fn iter(v: [uint], f: fn(uint)) {
    let i = 0u;
    let n = vec::len(v);
    while i < n {
        f(v[i]);
        i += 1u;
    }
}
