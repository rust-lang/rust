// run-pass

struct Foo<T>(T);
struct Bar<T> { x: T }
struct W(u32);
struct A { a: u32 }

const fn basics((a,): (u32,)) -> u32 {
    // Deferred assignment:
    let b: u32;
    b = a + 1;

    // Immediate assignment:
    let c: u32 = b + 1;

    // Mutables:
    let mut d: u32 = c + 1;
    d = d + 1;
    // +4 so far.

    // No effect statements work:
    ; ;
    1;

    // Array projection
    let mut arr: [u32; 1] = [0];
    arr[0] = 1;
    d = d + arr[0];
    // +5

    // Field projection:
    let mut foo: Foo<u32> = Foo(0);
    let mut bar: Bar<u32> = Bar { x: 0 };
    foo.0 = 1;
    bar.x = 1;
    d = d + foo.0 + bar.x;
    // +7

    // Array + Field projection:
    let mut arr: [Foo<u32>; 1] = [Foo(0)];
    arr[0].0 = 1;
    d = d + arr[0].0;
    let mut arr: [Bar<u32>; 1] = [Bar { x: 0 }];
    arr[0].x = 1;
    d = d + arr[0].x;
    // +9

    // Field + Array projection:
    let mut arr: Foo<[u32; 1]> = Foo([0]);
    (arr.0)[0] = 1;
    d = d + (arr.0)[0];
    let mut arr: Bar<[u32; 1]> = Bar { x: [0] };
    arr.x[0] = 1;
    d = d + arr.x[0];
    // +11

    d
}

const fn add_assign(W(a): W) -> u32 {
    // Mutables:
    let mut d: u32 = a + 1;
    d += 1;
    // +2 so far.

    // Array projection
    let mut arr: [u32; 1] = [0];
    arr[0] += 1;
    d += arr[0];
    // +3

    // Field projection:
    let mut foo: Foo<u32> = Foo(0);
    let mut bar: Bar<u32> = Bar { x: 0 };
    foo.0 += 1;
    bar.x += 1;
    d += foo.0 + bar.x;
    // +5

    // Array + Field projection:
    let mut arr: [Foo<u32>; 1] = [Foo(0)];
    arr[0].0 += 1;
    d += arr[0].0;
    let mut arr: [Bar<u32>; 1] = [Bar { x: 0 }];
    arr[0].x += 1;
    d += arr[0].x;
    // +7

    // Field + Array projection:
    let mut arr: Foo<[u32; 1]> = Foo([0]);
    (arr.0)[0] += 1;
    d += (arr.0)[0];
    let mut arr: Bar<[u32; 1]> = Bar { x: [0] };
    arr.x[0] += 1;
    d += arr.x[0];
    // +9

    d
}

const fn mul_assign(A { a }: A) -> u32 {
    // Mutables:
    let mut d: u32 = a + 1;
    d *= 2;
    // 2^1 * (a + 1)

    // Array projection
    let mut arr: [u32; 1] = [1];
    arr[0] *= 2;
    d *= arr[0];
    // 2^2 * (a + 1)

    // Field projection:
    let mut foo: Foo<u32> = Foo(1);
    let mut bar: Bar<u32> = Bar { x: 1 };
    foo.0 *= 2;
    bar.x *= 2;
    d *= foo.0 + bar.x;
    // 2^4 * (a + 1)

    // Array + Field projection:
    let mut arr: [Foo<u32>; 1] = [Foo(1)];
    arr[0].0 *= 2;
    d *= arr[0].0;
    let mut arr: [Bar<u32>; 1] = [Bar { x: 1 }];
    arr[0].x *= 2;
    d *= arr[0].x;
    // 2^6 * (a + 1)

    // Field + Array projection:
    let mut arr: Foo<[u32; 1]> = Foo([1]);
    (arr.0)[0] *= 2;
    d *= (arr.0)[0];
    let mut arr: Bar<[u32; 1]> = Bar { x: [1] };
    arr.x[0] *= 2;
    d *= arr.x[0];
    // 2^8 * (a + 1)

    d
}

const fn div_assign(a: [u32; 1]) -> u32 {
    let a = a[0];
    // Mutables:
    let mut d: u32 = 1024 * a;
    d /= 2;
    // 512

    // Array projection
    let mut arr: [u32; 1] = [4];
    arr[0] /= 2;
    d /= arr[0];
    // 256

    // Field projection:
    let mut foo: Foo<u32> = Foo(4);
    let mut bar: Bar<u32> = Bar { x: 4 };
    foo.0 /= 2;
    bar.x /= 2;
    d /= foo.0;
    d /= bar.x;
    // 64

    // Array + Field projection:
    let mut arr: [Foo<u32>; 1] = [Foo(4)];
    arr[0].0 /= 2;
    d /= arr[0].0;
    let mut arr: [Bar<u32>; 1] = [Bar { x: 4 }];
    arr[0].x /= 2;
    d /= arr[0].x;
    // 16

    // Field + Array projection:
    let mut arr: Foo<[u32; 1]> = Foo([4]);
    (arr.0)[0] /= 2;
    d /= (arr.0)[0];
    let mut arr: Bar<[u32; 1]> = Bar { x: [4] };
    arr.x[0] /= 2;
    d /= arr.x[0];
    // 4

    d
}

const fn rem_assign(W(a): W) -> u32 {
    // Mutables:
    let mut d: u32 = a;
    d %= 10;
    d += 10;

    // Array projection
    let mut arr: [u32; 1] = [3];
    arr[0] %= 2;
    d %= 9 + arr[0];
    d += 10;

    // Field projection:
    let mut foo: Foo<u32> = Foo(5);
    let mut bar: Bar<u32> = Bar { x: 7 };
    foo.0 %= 2;
    bar.x %= 2;
    d %= 8 + foo.0 + bar.x;
    d += 10;

    // Array + Field projection:
    let mut arr: [Foo<u32>; 1] = [Foo(4)];
    arr[0].0 %= 3;
    d %= 9 + arr[0].0;
    d += 10;
    let mut arr: [Bar<u32>; 1] = [Bar { x: 7 }];
    arr[0].x %= 3;
    d %= 9 + arr[0].x;
    d += 10;

    // Field + Array projection:
    let mut arr: Foo<[u32; 1]> = Foo([6]);
    (arr.0)[0] %= 5;
    d %= 9 + (arr.0)[0];
    let mut arr: Bar<[u32; 1]> = Bar { x: [11] };
    arr.x[0] %= 5;
    d %= 9 + arr.x[0];

    d
}

const fn sub_assign(W(a): W) -> u32 {
    // Mutables:
    let mut d: u32 = a;
    d -= 1;

    // Array projection
    let mut arr: [u32; 1] = [2];
    arr[0] -= 1;
    d -= arr[0];

    // Field projection:
    let mut foo: Foo<u32> = Foo(2);
    let mut bar: Bar<u32> = Bar { x: 2 };
    foo.0 -= 1;
    bar.x -= 1;
    d -= foo.0 + bar.x;

    // Array + Field projection:
    let mut arr: [Foo<u32>; 1] = [Foo(2)];
    arr[0].0 -= 1;
    d -= arr[0].0;
    let mut arr: [Bar<u32>; 1] = [Bar { x: 2 }];
    arr[0].x -= 1;
    d -= arr[0].x;

    // Field + Array projection:
    let mut arr: Foo<[u32; 1]> = Foo([2]);
    (arr.0)[0] -= 1;
    d -= (arr.0)[0];
    let mut arr: Bar<[u32; 1]> = Bar { x: [2] };
    arr.x[0] -= 1;
    d -= arr.x[0];

    d
}

const fn shl_assign(W(a): W) -> u32 {
    // Mutables:
    let mut d: u32 = a;
    d <<= 1; // 10

    // Array projection
    let mut arr: [u32; 1] = [1];
    arr[0] <<= 1;
    d <<= arr[0]; // 10 << 2

    // Field projection:
    let mut foo: Foo<u32> = Foo(1);
    let mut bar: Bar<u32> = Bar { x: 1 };
    foo.0 <<= 1;
    bar.x <<= 1;
    d <<= foo.0 + bar.x; // 1000 << 4

    // Array + Field projection:
    let mut arr: [Foo<u32>; 1] = [Foo(1)];
    arr[0].0 <<= 1;
    d <<= arr[0].0; // 1000_0000 << 2
    let mut arr: [Bar<u32>; 1] = [Bar { x: 1 }];
    arr[0].x <<= 1;
    d <<= arr[0].x; // 1000_0000_00 << 2

    // Field + Array projection:
    let mut arr: Foo<[u32; 1]> = Foo([1]);
    (arr.0)[0] <<= 1;
    d <<= (arr.0)[0]; // 1000_0000_0000 << 2
    let mut arr: Bar<[u32; 1]> = Bar { x: [1] };
    arr.x[0] <<= 1;
    d <<= arr.x[0]; // 1000_0000_0000_00 << 2

    d
}

const fn shr_assign(W(a): W) -> u32 {
    // Mutables:
    let mut d: u32 = a;
    d >>= 1; // /= 2

    // Array projection
    let mut arr: [u32; 1] = [2];
    arr[0] >>= 1;
    d >>= arr[0]; // /= 4

    // Field projection:
    let mut foo: Foo<u32> = Foo(2);
    let mut bar: Bar<u32> = Bar { x: 2 };
    foo.0 >>= 1;
    bar.x >>= 1;
    d >>= foo.0 + bar.x; // /= 16

    // Array + Field projection:
    let mut arr: [Foo<u32>; 1] = [Foo(2)];
    arr[0].0 >>= 1;
    d >>= arr[0].0; // /= 32
    let mut arr: [Bar<u32>; 1] = [Bar { x: 2 }];
    arr[0].x >>= 1;
    d >>= arr[0].x; // /= 64

    // Field + Array projection:
    let mut arr: Foo<[u32; 1]> = Foo([2]);
    (arr.0)[0] >>= 1;
    d >>= (arr.0)[0]; // /= 128
    let mut arr: Bar<[u32; 1]> = Bar { x: [2] };
    arr.x[0] >>= 1;
    d >>= arr.x[0]; // /= 256

    d
}

const fn bit_and_assign(W(a): W) -> u32 {
    let f = 0b1111_1111_1111_1111;

    // Mutables:
    let mut d: u32 = a;
    d &= 0b1111_1111_1111_1110;

    // Array projection
    let mut arr: [u32; 1] = [f];
    arr[0] &= 0b1111_1111_1111_1101;
    d &= arr[0];

    // Field projection:
    let mut foo: Foo<u32> = Foo(f);
    let mut bar: Bar<u32> = Bar { x: f };
    foo.0 &= 0b1111_1111_1111_0111;
    bar.x &= 0b1111_1111_1101_1111;
    d &= foo.0 & bar.x;

    // Array + Field projection:
    let mut arr: [Foo<u32>; 1] = [Foo(f)];
    arr[0].0 &= 0b1111_1110_1111_1111;
    d &= arr[0].0;
    let mut arr: [Bar<u32>; 1] = [Bar { x: f }];
    arr[0].x &= 0b1111_1101_1111_1111;
    d &= arr[0].x;

    // Field + Array projection:
    let mut arr: Foo<[u32; 1]> = Foo([f]);
    (arr.0)[0] &= 0b1011_1111_1111_1111;
    d &= (arr.0)[0];
    let mut arr: Bar<[u32; 1]> = Bar { x: [f] };
    arr.x[0] &= 0b0111_1111_1111_1111;
    d &= arr.x[0];

    d
}

const fn bit_or_assign(W(a): W) -> u32 {
    let f = 0b0000_0000_0000_0000;

    // Mutables:
    let mut d: u32 = a;
    d |= 0b0000_0000_0000_0001;

    // Array projection
    let mut arr: [u32; 1] = [f];
    arr[0] |= 0b0000_0000_0000_1001;
    d |= arr[0];

    // Field projection:
    let mut foo: Foo<u32> = Foo(f);
    let mut bar: Bar<u32> = Bar { x: f };
    foo.0 |= 0b0000_0000_0001_0000;
    bar.x |= 0b0000_0000_0100_0000;
    d |= foo.0 | bar.x;

    // Array + Field projection:
    let mut arr: [Foo<u32>; 1] = [Foo(f)];
    arr[0].0 |= 0b0000_0001_0000_0000;
    d |= arr[0].0;
    let mut arr: [Bar<u32>; 1] = [Bar { x: f }];
    arr[0].x |= 0b0000_0010_0000_0000;
    d |= arr[0].x;

    // Field + Array projection:
    let mut arr: Foo<[u32; 1]> = Foo([f]);
    (arr.0)[0] |= 0b1000_0000_0000_0000;
    d |= (arr.0)[0]; // /= 128
    let mut arr: Bar<[u32; 1]> = Bar { x: [f] };
    arr.x[0] |= 0b1100_0000_0000_0000;
    d |= arr.x[0]; // /= 256

    d
}

const fn bit_xor_assign(W(a): W) -> u32 {
    let f = 0b0000_0000_0000_0000;

    // Mutables:
    let mut d: u32 = a;
    d ^= 0b0000_0000_0000_0001;

    // Array projection
    let mut arr: [u32; 1] = [f];
    arr[0] ^= 0b0000_0000_0000_0010;
    d ^= arr[0];

    // Field projection:
    let mut foo: Foo<u32> = Foo(f);
    let mut bar: Bar<u32> = Bar { x: f };
    foo.0 ^= 0b0000_0000_0001_0000;
    bar.x ^= 0b0000_0000_1000_0000;
    d ^= foo.0 ^ bar.x;

    // Array + Field projection:
    let mut arr: [Foo<u32>; 1] = [Foo(f)];
    arr[0].0 ^= 0b0000_0001_0000_0000;
    d ^= arr[0].0;
    let mut arr: [Bar<u32>; 1] = [Bar { x: f }];
    arr[0].x ^= 0b0000_0010_0000_0000;
    d ^= arr[0].x;

    // Field + Array projection:
    let mut arr: Foo<[u32; 1]> = Foo([f]);
    (arr.0)[0] ^= 0b0100_0000_0000_0000;
    d ^= (arr.0)[0];
    let mut arr: Bar<[u32; 1]> = Bar { x: [f] };
    arr.x[0] ^= 0b1000_0000_0000_0000;
    d ^= arr.x[0];

    d
}

macro_rules! test {
    ($c:ident, $e:expr, $r:expr) => {
        const $c: u32 = $e;
        assert_eq!($c, $r);
        assert_eq!($e, $r);
    }
}

fn main() {
    test!(BASICS, basics((2,)), 13);
    test!(ADD, add_assign(W(1)), 10);
    test!(MUL, mul_assign(A { a: 0 }), 256);
    test!(DIV, div_assign([1]), 4);
    test!(REM, rem_assign(W(5)), 5);
    test!(SUB, sub_assign(W(8)), 0);
    test!(SHL, shl_assign(W(1)), 0b1000_0000_0000_0000);
    test!(SHR, shr_assign(W(256)), 1);
    test!(AND, bit_and_assign(W(0b1011_1111_1111_1111_1111)), 0b0011_1100_1101_0100);
    test!(OR, bit_or_assign(W(0b1011_0000_0000_0000)), 0b1111_0011_0101_1001);
    test!(XOR, bit_xor_assign(W(0b0000_0000_0000_0000)), 0b1100_0011_1001_0011);
}
