#[inline]
fn iter<T>(v: [T], f: fn(T)) {
    let i = 0u;
    let n = vec::len(v);
    while i < n {
        f(v[i]);
        i += 1u;
    }
}
