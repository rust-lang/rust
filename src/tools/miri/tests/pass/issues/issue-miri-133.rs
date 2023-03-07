use std::mem::size_of;

struct S<U, V> {
    _u: U,
    size_of_u: usize,
    _v: V,
    size_of_v: usize,
}

impl<U, V> S<U, V> {
    fn new(u: U, v: V) -> Self {
        S { _u: u, size_of_u: size_of::<U>(), _v: v, size_of_v: size_of::<V>() }
    }
}

impl<V, U> Drop for S<U, V> {
    fn drop(&mut self) {
        assert_eq!(size_of::<U>(), self.size_of_u);
        assert_eq!(size_of::<V>(), self.size_of_v);
    }
}

fn main() {
    S::new(0u8, 1u16);
}
