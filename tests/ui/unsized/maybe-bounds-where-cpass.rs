//@ check-pass

#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

struct S<T>(*const T);


fn main() {
    let u = vec![1, 2, 3];
    let _s: S<[u8]> = S(&u[..]);
}
