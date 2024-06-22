//@ build-fail
//@ ignore-32bit
//@ error-pattern: evaluation of `<S<u8> as std::mem::SizedTypeProperties>::SIZE_IN_BYTES` failed

struct S<T> {
    x: [T; !0],
}

pub fn f() -> usize {
    std::mem::size_of::<S<u8>>()
}

fn main() {
    let x = f();
}
