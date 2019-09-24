// run-pass
// ignore-emscripten no i128 support

#![feature(intrinsics)]

mod rusti {
    extern "rust-intrinsic" {
        pub fn ctpop<T>(x: T) -> T;
        pub fn ctlz<T>(x: T) -> T;
        pub fn ctlz_nonzero<T>(x: T) -> T;
        pub fn cttz<T>(x: T) -> T;
        pub fn cttz_nonzero<T>(x: T) -> T;
        pub fn bswap<T>(x: T) -> T;
        pub fn bitreverse<T>(x: T) -> T;
    }
}

pub fn main() {
    use rusti::*;

    assert_eq!(ctpop(0u8), 0); assert_eq!(ctpop(0i8), 0);
    assert_eq!(ctpop(0u16), 0); assert_eq!(ctpop(0i16), 0);
    assert_eq!(ctpop(0u32), 0); assert_eq!(ctpop(0i32), 0);
    assert_eq!(ctpop(0u64), 0); assert_eq!(ctpop(0i64), 0);
    assert_eq!(ctpop(0u128), 0); assert_eq!(ctpop(0i128), 0);

    assert_eq!(ctpop(1u8), 1); assert_eq!(ctpop(1i8), 1);
    assert_eq!(ctpop(1u16), 1); assert_eq!(ctpop(1i16), 1);
    assert_eq!(ctpop(1u32), 1); assert_eq!(ctpop(1i32), 1);
    assert_eq!(ctpop(1u64), 1); assert_eq!(ctpop(1i64), 1);
    assert_eq!(ctpop(1u128), 1); assert_eq!(ctpop(1i128), 1);

    assert_eq!(ctpop(10u8), 2); assert_eq!(ctpop(10i8), 2);
    assert_eq!(ctpop(10u16), 2); assert_eq!(ctpop(10i16), 2);
    assert_eq!(ctpop(10u32), 2); assert_eq!(ctpop(10i32), 2);
    assert_eq!(ctpop(10u64), 2); assert_eq!(ctpop(10i64), 2);
    assert_eq!(ctpop(10u128), 2); assert_eq!(ctpop(10i128), 2);

    assert_eq!(ctpop(100u8), 3); assert_eq!(ctpop(100i8), 3);
    assert_eq!(ctpop(100u16), 3); assert_eq!(ctpop(100i16), 3);
    assert_eq!(ctpop(100u32), 3); assert_eq!(ctpop(100i32), 3);
    assert_eq!(ctpop(100u64), 3); assert_eq!(ctpop(100i64), 3);
    assert_eq!(ctpop(100u128), 3); assert_eq!(ctpop(100i128), 3);

    assert_eq!(ctpop(-1i8 as u8), 8); assert_eq!(ctpop(-1i8), 8);
    assert_eq!(ctpop(-1i16 as u16), 16); assert_eq!(ctpop(-1i16), 16);
    assert_eq!(ctpop(-1i32 as u32), 32); assert_eq!(ctpop(-1i32), 32);
    assert_eq!(ctpop(-1i64 as u64), 64); assert_eq!(ctpop(-1i64), 64);
    assert_eq!(ctpop(-1i128 as u128), 128); assert_eq!(ctpop(-1i128), 128);

    assert_eq!(ctlz(0u8), 8); assert_eq!(ctlz(0i8), 8);
    assert_eq!(ctlz(0u16), 16); assert_eq!(ctlz(0i16), 16);
    assert_eq!(ctlz(0u32), 32); assert_eq!(ctlz(0i32), 32);
    assert_eq!(ctlz(0u64), 64); assert_eq!(ctlz(0i64), 64);
    assert_eq!(ctlz(0u128), 128); assert_eq!(ctlz(0i128), 128);

    assert_eq!(ctlz(1u8), 7); assert_eq!(ctlz(1i8), 7);
    assert_eq!(ctlz(1u16), 15); assert_eq!(ctlz(1i16), 15);
    assert_eq!(ctlz(1u32), 31); assert_eq!(ctlz(1i32), 31);
    assert_eq!(ctlz(1u64), 63); assert_eq!(ctlz(1i64), 63);
    assert_eq!(ctlz(1u128), 127); assert_eq!(ctlz(1i128), 127);

    assert_eq!(ctlz(10u8), 4); assert_eq!(ctlz(10i8), 4);
    assert_eq!(ctlz(10u16), 12); assert_eq!(ctlz(10i16), 12);
    assert_eq!(ctlz(10u32), 28); assert_eq!(ctlz(10i32), 28);
    assert_eq!(ctlz(10u64), 60); assert_eq!(ctlz(10i64), 60);
    assert_eq!(ctlz(10u128), 124); assert_eq!(ctlz(10i128), 124);

    assert_eq!(ctlz(100u8), 1); assert_eq!(ctlz(100i8), 1);
    assert_eq!(ctlz(100u16), 9); assert_eq!(ctlz(100i16), 9);
    assert_eq!(ctlz(100u32), 25); assert_eq!(ctlz(100i32), 25);
    assert_eq!(ctlz(100u64), 57); assert_eq!(ctlz(100i64), 57);
    assert_eq!(ctlz(100u128), 121); assert_eq!(ctlz(100i128), 121);

    unsafe {
        assert_eq!(ctlz_nonzero(1u8), 7); assert_eq!(ctlz_nonzero(1i8), 7);
        assert_eq!(ctlz_nonzero(1u16), 15); assert_eq!(ctlz_nonzero(1i16), 15);
        assert_eq!(ctlz_nonzero(1u32), 31); assert_eq!(ctlz_nonzero(1i32), 31);
        assert_eq!(ctlz_nonzero(1u64), 63); assert_eq!(ctlz_nonzero(1i64), 63);
        assert_eq!(ctlz_nonzero(1u128), 127); assert_eq!(ctlz_nonzero(1i128), 127);

        assert_eq!(ctlz_nonzero(10u8), 4); assert_eq!(ctlz_nonzero(10i8), 4);
        assert_eq!(ctlz_nonzero(10u16), 12); assert_eq!(ctlz_nonzero(10i16), 12);
        assert_eq!(ctlz_nonzero(10u32), 28); assert_eq!(ctlz_nonzero(10i32), 28);
        assert_eq!(ctlz_nonzero(10u64), 60); assert_eq!(ctlz_nonzero(10i64), 60);
        assert_eq!(ctlz_nonzero(10u128), 124); assert_eq!(ctlz_nonzero(10i128), 124);

        assert_eq!(ctlz_nonzero(100u8), 1); assert_eq!(ctlz_nonzero(100i8), 1);
        assert_eq!(ctlz_nonzero(100u16), 9); assert_eq!(ctlz_nonzero(100i16), 9);
        assert_eq!(ctlz_nonzero(100u32), 25); assert_eq!(ctlz_nonzero(100i32), 25);
        assert_eq!(ctlz_nonzero(100u64), 57); assert_eq!(ctlz_nonzero(100i64), 57);
        assert_eq!(ctlz_nonzero(100u128), 121); assert_eq!(ctlz_nonzero(100i128), 121);
    }

    assert_eq!(cttz(-1i8 as u8), 0); assert_eq!(cttz(-1i8), 0);
    assert_eq!(cttz(-1i16 as u16), 0); assert_eq!(cttz(-1i16), 0);
    assert_eq!(cttz(-1i32 as u32), 0); assert_eq!(cttz(-1i32), 0);
    assert_eq!(cttz(-1i64 as u64), 0); assert_eq!(cttz(-1i64), 0);
    assert_eq!(cttz(-1i128 as u128), 0); assert_eq!(cttz(-1i128), 0);

    assert_eq!(cttz(0u8), 8); assert_eq!(cttz(0i8), 8);
    assert_eq!(cttz(0u16), 16); assert_eq!(cttz(0i16), 16);
    assert_eq!(cttz(0u32), 32); assert_eq!(cttz(0i32), 32);
    assert_eq!(cttz(0u64), 64); assert_eq!(cttz(0i64), 64);
    assert_eq!(cttz(0u128), 128); assert_eq!(cttz(0i128), 128);

    assert_eq!(cttz(1u8), 0); assert_eq!(cttz(1i8), 0);
    assert_eq!(cttz(1u16), 0); assert_eq!(cttz(1i16), 0);
    assert_eq!(cttz(1u32), 0); assert_eq!(cttz(1i32), 0);
    assert_eq!(cttz(1u64), 0); assert_eq!(cttz(1i64), 0);
    assert_eq!(cttz(1u128), 0); assert_eq!(cttz(1i128), 0);

    assert_eq!(cttz(10u8), 1); assert_eq!(cttz(10i8), 1);
    assert_eq!(cttz(10u16), 1); assert_eq!(cttz(10i16), 1);
    assert_eq!(cttz(10u32), 1); assert_eq!(cttz(10i32), 1);
    assert_eq!(cttz(10u64), 1); assert_eq!(cttz(10i64), 1);
    assert_eq!(cttz(10u128), 1); assert_eq!(cttz(10i128), 1);

    assert_eq!(cttz(100u8), 2); assert_eq!(cttz(100i8), 2);
    assert_eq!(cttz(100u16), 2); assert_eq!(cttz(100i16), 2);
    assert_eq!(cttz(100u32), 2); assert_eq!(cttz(100i32), 2);
    assert_eq!(cttz(100u64), 2); assert_eq!(cttz(100i64), 2);
    assert_eq!(cttz(100u128), 2); assert_eq!(cttz(100i128), 2);

    unsafe {
        assert_eq!(cttz_nonzero(-1i8 as u8), 0); assert_eq!(cttz_nonzero(-1i8), 0);
        assert_eq!(cttz_nonzero(-1i16 as u16), 0); assert_eq!(cttz_nonzero(-1i16), 0);
        assert_eq!(cttz_nonzero(-1i32 as u32), 0); assert_eq!(cttz_nonzero(-1i32), 0);
        assert_eq!(cttz_nonzero(-1i64 as u64), 0); assert_eq!(cttz_nonzero(-1i64), 0);
        assert_eq!(cttz_nonzero(-1i128 as u128), 0); assert_eq!(cttz_nonzero(-1i128), 0);

        assert_eq!(cttz_nonzero(1u8), 0); assert_eq!(cttz_nonzero(1i8), 0);
        assert_eq!(cttz_nonzero(1u16), 0); assert_eq!(cttz_nonzero(1i16), 0);
        assert_eq!(cttz_nonzero(1u32), 0); assert_eq!(cttz_nonzero(1i32), 0);
        assert_eq!(cttz_nonzero(1u64), 0); assert_eq!(cttz_nonzero(1i64), 0);
        assert_eq!(cttz_nonzero(1u128), 0); assert_eq!(cttz_nonzero(1i128), 0);

        assert_eq!(cttz_nonzero(10u8), 1); assert_eq!(cttz_nonzero(10i8), 1);
        assert_eq!(cttz_nonzero(10u16), 1); assert_eq!(cttz_nonzero(10i16), 1);
        assert_eq!(cttz_nonzero(10u32), 1); assert_eq!(cttz_nonzero(10i32), 1);
        assert_eq!(cttz_nonzero(10u64), 1); assert_eq!(cttz_nonzero(10i64), 1);
        assert_eq!(cttz_nonzero(10u128), 1); assert_eq!(cttz_nonzero(10i128), 1);

        assert_eq!(cttz_nonzero(100u8), 2); assert_eq!(cttz_nonzero(100i8), 2);
        assert_eq!(cttz_nonzero(100u16), 2); assert_eq!(cttz_nonzero(100i16), 2);
        assert_eq!(cttz_nonzero(100u32), 2); assert_eq!(cttz_nonzero(100i32), 2);
        assert_eq!(cttz_nonzero(100u64), 2); assert_eq!(cttz_nonzero(100i64), 2);
        assert_eq!(cttz_nonzero(100u128), 2); assert_eq!(cttz_nonzero(100i128), 2);
    }

    assert_eq!(bswap(0x0Au8), 0x0A); // no-op
    assert_eq!(bswap(0x0Ai8), 0x0A); // no-op
    assert_eq!(bswap(0x0A0Bu16), 0x0B0A);
    assert_eq!(bswap(0x0A0Bi16), 0x0B0A);
    assert_eq!(bswap(0x0ABBCC0Du32), 0x0DCCBB0A);
    assert_eq!(bswap(0x0ABBCC0Di32), 0x0DCCBB0A);
    assert_eq!(bswap(0x0122334455667708u64), 0x0877665544332201);
    assert_eq!(bswap(0x0122334455667708i64), 0x0877665544332201);
    assert_eq!(bswap(0x0122334455667708u128), 0x08776655443322010000000000000000);
    assert_eq!(bswap(0x0122334455667708i128), 0x08776655443322010000000000000000);

    assert_eq!(bitreverse(0x0Au8), 0x50);
    assert_eq!(bitreverse(0x0Ai8), 0x50);
    assert_eq!(bitreverse(0x0A0Cu16), 0x3050);
    assert_eq!(bitreverse(0x0A0Ci16), 0x3050);
    assert_eq!(bitreverse(0x0ABBCC0Eu32), 0x7033DD50);
    assert_eq!(bitreverse(0x0ABBCC0Ei32), 0x7033DD50);
    assert_eq!(bitreverse(0x0122334455667708u64), 0x10EE66AA22CC4480);
    assert_eq!(bitreverse(0x0122334455667708i64), 0x10EE66AA22CC4480);
    assert_eq!(bitreverse(0x0122334455667708u128), 0x10EE66AA22CC44800000000000000000);
    assert_eq!(bitreverse(0x0122334455667708i128), 0x10EE66AA22CC44800000000000000000);
}
