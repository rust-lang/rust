//~? ERROR values of the type `[u8; usize::MAX]` are too big for the target architecture

//@ build-fail
//@ ignore-32bit

struct S<T> {
    x: [T; !0],
}

pub fn f() -> usize {
    std::mem::size_of::<S<u8>>()
}

fn main() {
    let x = f();
}
