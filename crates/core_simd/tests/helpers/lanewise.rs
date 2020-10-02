pub fn apply_unary_lanewise<T: Copy, V: AsMut<[T]> + Default>(mut x: V, f: impl Fn(T) -> T) -> V {
    for lane in x.as_mut() {
        *lane = f(*lane)
    }
    x
}

pub fn apply_binary_lanewise<T: Copy, V: AsRef<[T]> + AsMut<[T]> + Default>(
    a: V,
    b: V,
    f: impl Fn(T, T) -> T,
) -> V {
    let mut out = V::default();
    let out_slice = out.as_mut();
    let a_slice = a.as_ref();
    let b_slice = b.as_ref();
    for (o, (a, b)) in out_slice.iter_mut().zip(a_slice.iter().zip(b_slice.iter())) {
        *o = f(*a, *b);
    }
    out
}

pub fn apply_binary_scalar_rhs_lanewise<T: Copy, V: AsRef<[T]> + AsMut<[T]> + Default>(
    a: V,
    b: T,
    f: impl Fn(T, T) -> T,
) -> V {
    let mut out = V::default();
    let out_slice = out.as_mut();
    let a_slice = a.as_ref();
    for (o, a) in out_slice.iter_mut().zip(a_slice.iter()) {
        *o = f(*a, b);
    }
    out
}

pub fn apply_binary_scalar_lhs_lanewise<T: Copy, V: AsRef<[T]> + AsMut<[T]> + Default>(
    a: T,
    b: V,
    f: impl Fn(T, T) -> T,
) -> V {
    let mut out = V::default();
    let out_slice = out.as_mut();
    let b_slice = b.as_ref();
    for (o, b) in out_slice.iter_mut().zip(b_slice.iter()) {
        *o = f(a, *b);
    }
    out
}
