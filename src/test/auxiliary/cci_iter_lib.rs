#[link(name="cci_iter_lib", vers="0.0")];

#[inline]
fn iter<T>(v: [T], f: fn(T)) {
    let mut i = 0u;
    let n = vec::len(v);
    while i < n {
        f(v[i]);
        i += 1u;
    }
}
