#![feature(rustc_attrs)]

struct S<T>(*const T) where T: ?Sized;

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let u = vec![1, 2, 3];
    let _s: S<[u8]> = S(&u[..]);
}
