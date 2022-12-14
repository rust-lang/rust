#[repr(C)]
#[derive(Clone, Copy)]
struct Pair<T, U>(T, U);
#[repr(C)]
#[derive(Clone, Copy)]
struct Triple<T>(T, T, T);

#[repr(C)]
union U<A: Copy, B: Copy> {
    a: Pair<A, A>,
    b: B,
}

#[repr(C)]
union W<A: Copy, B: Copy> {
    a: A,
    b: B,
}

#[cfg(target_endian = "little")]
unsafe fn check() {
    let mut u = U::<u8, u16> { b: 0xDE_DE };
    u.a.0 = 0xBE;
    assert_eq!(u.b, 0xDE_BE);

    let mut u = U::<u16, u32> { b: 0xDEAD_DEAD };
    u.a.0 = 0xBEEF;
    assert_eq!(u.b, 0xDEAD_BEEF);

    let mut u = U::<u32, u64> { b: 0xDEADBEEF_DEADBEEF };
    u.a.0 = 0xBAADF00D;
    assert_eq!(u.b, 0xDEADBEEF_BAADF00D);

    let mut w = W::<Pair<Triple<u8>, u8>, u32> { b: 0xDEAD_DEAD };
    w.a.0 = Triple(0, 0, 0);
    assert_eq!(w.b, 0xDE00_0000);

    let mut w = W::<Pair<u8, Triple<u8>>, u32> { b: 0xDEAD_DEAD };
    w.a.1 = Triple(0, 0, 0);
    assert_eq!(w.b, 0x0000_00AD);
}

#[cfg(target_endian = "big")]
unsafe fn check() {
    let mut u = U::<u8, u16> { b: 0xDE_DE };
    u.a.0 = 0xBE;
    assert_eq!(u.b, 0xBE_DE);

    let mut u = U::<u16, u32> { b: 0xDEAD_DEAD };
    u.a.0 = 0xBEEF;
    assert_eq!(u.b, 0xBEEF_DEAD);

    let mut u = U::<u32, u64> { b: 0xDEADBEEF_DEADBEEF };
    u.a.0 = 0xBAADF00D;
    assert_eq!(u.b, 0xBAADF00D_DEADBEEF);

    let mut w = W::<Pair<Triple<u8>, u8>, u32> { b: 0xDEAD_DEAD };
    w.a.0 = Triple(0, 0, 0);
    assert_eq!(w.b, 0x0000_00AD);

    let mut w = W::<Pair<u8, Triple<u8>>, u32> { b: 0xDEAD_DEAD };
    w.a.1 = Triple(0, 0, 0);
    assert_eq!(w.b, 0xDE00_0000);
}

fn main() {
    unsafe {
        check();
    }
}
