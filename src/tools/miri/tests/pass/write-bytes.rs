#![feature(core_intrinsics)] // for `volatile_set_memory`

#[repr(C)]
#[derive(Copy, Clone)]
struct Foo {
    a: u64,
    b: u64,
    c: u64,
}

fn main() {
    const LENGTH: usize = 10;
    let mut v: [u64; LENGTH] = [0; LENGTH];

    for idx in 0..LENGTH {
        assert_eq!(v[idx], 0);
    }

    unsafe {
        let p = v.as_mut_ptr();
        ::std::ptr::write_bytes(p, 0xab, LENGTH);
    }

    for idx in 0..LENGTH {
        assert_eq!(v[idx], 0xabababababababab);
    }

    // -----

    let mut w: [Foo; LENGTH] = [Foo { a: 0, b: 0, c: 0 }; LENGTH];
    for idx in 0..LENGTH {
        assert_eq!(w[idx].a, 0);
        assert_eq!(w[idx].b, 0);
        assert_eq!(w[idx].c, 0);
    }

    unsafe {
        let p = w.as_mut_ptr();
        ::std::ptr::write_bytes(p, 0xcd, LENGTH);
    }

    for idx in 0..LENGTH {
        assert_eq!(w[idx].a, 0xcdcdcdcdcdcdcdcd);
        assert_eq!(w[idx].b, 0xcdcdcdcdcdcdcdcd);
        assert_eq!(w[idx].c, 0xcdcdcdcdcdcdcdcd);
    }

    // -----
    // `std::intrinsics::volatile_set_memory` should behave identically

    let mut v: [u64; LENGTH] = [0; LENGTH];

    for idx in 0..LENGTH {
        assert_eq!(v[idx], 0);
    }

    unsafe {
        let p = v.as_mut_ptr();
        ::std::intrinsics::volatile_set_memory(p, 0xab, LENGTH);
    }

    for idx in 0..LENGTH {
        assert_eq!(v[idx], 0xabababababababab);
    }

    // -----

    let mut w: [Foo; LENGTH] = [Foo { a: 0, b: 0, c: 0 }; LENGTH];
    for idx in 0..LENGTH {
        assert_eq!(w[idx].a, 0);
        assert_eq!(w[idx].b, 0);
        assert_eq!(w[idx].c, 0);
    }

    unsafe {
        let p = w.as_mut_ptr();
        ::std::intrinsics::volatile_set_memory(p, 0xcd, LENGTH);
    }

    for idx in 0..LENGTH {
        assert_eq!(w[idx].a, 0xcdcdcdcdcdcdcdcd);
        assert_eq!(w[idx].b, 0xcdcdcdcdcdcdcdcd);
        assert_eq!(w[idx].c, 0xcdcdcdcdcdcdcdcd);
    }
}
