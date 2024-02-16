//@ check-pass

struct S<T>(*const T) where T: ?Sized;


fn main() {
    let u = vec![1, 2, 3];
    let _s: S<[u8]> = S(&u[..]);
}
