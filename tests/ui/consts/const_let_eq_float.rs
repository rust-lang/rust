//@ run-pass

struct Foo<T>(T);
struct Bar<T> { x: T }
struct W(f32);
struct A { a: f32 }

#[allow(redundant_semicolons)]
const fn basics((a,): (f32,)) -> f32 {
    // Deferred assignment:
    let b: f32;
    b = a + 1.0;

    // Immediate assignment:
    let c: f32 = b + 1.0;

    // Mutables:
    let mut d: f32 = c + 1.0;
    d = d + 1.0;
    // +4 so far.

    // No effect statements work:
    ; ;
    1;

    // Array projection
    let mut arr: [f32; 1] = [0.0];
    arr[0] = 1.0;
    d = d + arr[0];
    // +5

    // Field projection:
    let mut foo: Foo<f32> = Foo(0.0);
    let mut bar: Bar<f32> = Bar { x: 0.0 };
    foo.0 = 1.0;
    bar.x = 1.0;
    d = d + foo.0 + bar.x;
    // +7

    // Array + Field projection:
    let mut arr: [Foo<f32>; 1] = [Foo(0.0)];
    arr[0].0 = 1.0;
    d = d + arr[0].0;
    let mut arr: [Bar<f32>; 1] = [Bar { x: 0.0 }];
    arr[0].x = 1.0;
    d = d + arr[0].x;
    // +9

    // Field + Array projection:
    let mut arr: Foo<[f32; 1]> = Foo([0.0]);
    (arr.0)[0] = 1.0;
    d = d + (arr.0)[0];
    let mut arr: Bar<[f32; 1]> = Bar { x: [0.0] };
    arr.x[0] = 1.0;
    d = d + arr.x[0];
    // +11

    d
}

const fn add_assign(W(a): W) -> f32 {
    // Mutables:
    let mut d: f32 = a + 1.0;
    d += 1.0;
    // +2 so far.

    // Array projection
    let mut arr: [f32; 1] = [0.0];
    arr[0] += 1.0;
    d += arr[0];
    // +3

    // Field projection:
    let mut foo: Foo<f32> = Foo(0.0);
    let mut bar: Bar<f32> = Bar { x: 0.0 };
    foo.0 += 1.0;
    bar.x += 1.0;
    d += foo.0 + bar.x;
    // +5

    // Array + Field projection:
    let mut arr: [Foo<f32>; 1] = [Foo(0.0)];
    arr[0].0 += 1.0;
    d += arr[0].0;
    let mut arr: [Bar<f32>; 1] = [Bar { x: 0.0 }];
    arr[0].x += 1.0;
    d += arr[0].x;
    // +7

    // Field + Array projection:
    let mut arr: Foo<[f32; 1]> = Foo([0.0]);
    (arr.0)[0] += 1.0;
    d += (arr.0)[0];
    let mut arr: Bar<[f32; 1]> = Bar { x: [0.0] };
    arr.x[0] += 1.0;
    d += arr.x[0];
    // +9

    d
}

const fn mul_assign(A { a }: A) -> f32 {
    // Mutables:
    let mut d: f32 = a + 1.0;
    d *= 2.0;
    // 2^1 * (a + 1)

    // Array projection
    let mut arr: [f32; 1] = [1.0];
    arr[0] *= 2.0;
    d *= arr[0];
    // 2^2 * (a + 1)

    // Field projection:
    let mut foo: Foo<f32> = Foo(1.0);
    let mut bar: Bar<f32> = Bar { x: 1.0 };
    foo.0 *= 2.0;
    bar.x *= 2.0;
    d *= foo.0 + bar.x;
    // 2^4 * (a + 1)

    // Array + Field projection:
    let mut arr: [Foo<f32>; 1] = [Foo(1.0)];
    arr[0].0 *= 2.0;
    d *= arr[0].0;
    let mut arr: [Bar<f32>; 1] = [Bar { x: 1.0 }];
    arr[0].x *= 2.0;
    d *= arr[0].x;
    // 2^6 * (a + 1)

    // Field + Array projection:
    let mut arr: Foo<[f32; 1]> = Foo([1.0]);
    (arr.0)[0] *= 2.0;
    d *= (arr.0)[0];
    let mut arr: Bar<[f32; 1]> = Bar { x: [1.0] };
    arr.x[0] *= 2.0;
    d *= arr.x[0];
    // 2^8 * (a + 1)

    d
}

const fn div_assign(a: [f32; 1]) -> f32 {
    let a = a[0];
    // Mutables:
    let mut d: f32 = 1024.0 * a;
    d /= 2.0;
    // 512

    // Array projection
    let mut arr: [f32; 1] = [4.0];
    arr[0] /= 2.0;
    d /= arr[0];
    // 256

    // Field projection:
    let mut foo: Foo<f32> = Foo(4.0);
    let mut bar: Bar<f32> = Bar { x: 4.0 };
    foo.0 /= 2.0;
    bar.x /= 2.0;
    d /= foo.0;
    d /= bar.x;
    // 64

    // Array + Field projection:
    let mut arr: [Foo<f32>; 1] = [Foo(4.0)];
    arr[0].0 /= 2.0;
    d /= arr[0].0;
    let mut arr: [Bar<f32>; 1] = [Bar { x: 4.0 }];
    arr[0].x /= 2.0;
    d /= arr[0].x;
    // 16

    // Field + Array projection:
    let mut arr: Foo<[f32; 1]> = Foo([4.0]);
    (arr.0)[0] /= 2.0;
    d /= (arr.0)[0];
    let mut arr: Bar<[f32; 1]> = Bar { x: [4.0] };
    arr.x[0] /= 2.0;
    d /= arr.x[0];
    // 4

    d
}

const fn rem_assign(W(a): W) -> f32 {
    // Mutables:
    let mut d: f32 = a;
    d %= 10.0;
    d += 10.0;

    // Array projection
    let mut arr: [f32; 1] = [3.0];
    arr[0] %= 2.0;
    d %= 9.0 + arr[0];
    d += 10.0;

    // Field projection:
    let mut foo: Foo<f32> = Foo(5.0);
    let mut bar: Bar<f32> = Bar { x: 7.0 };
    foo.0 %= 2.0;
    bar.x %= 2.0;
    d %= 8.0 + foo.0 + bar.x;
    d += 10.0;

    // Array + Field projection:
    let mut arr: [Foo<f32>; 1] = [Foo(4.0)];
    arr[0].0 %= 3.0;
    d %= 9.0 + arr[0].0;
    d += 10.0;
    let mut arr: [Bar<f32>; 1] = [Bar { x: 7.0 }];
    arr[0].x %= 3.0;
    d %= 9.0 + arr[0].x;
    d += 10.0;

    // Field + Array projection:
    let mut arr: Foo<[f32; 1]> = Foo([6.0]);
    (arr.0)[0] %= 5.0;
    d %= 9.0 + (arr.0)[0];
    let mut arr: Bar<[f32; 1]> = Bar { x: [11.0] };
    arr.x[0] %= 5.0;
    d %= 9.0 + arr.x[0];

    d
}

const fn sub_assign(W(a): W) -> f32 {
    // Mutables:
    let mut d: f32 = a;
    d -= 1.0;

    // Array projection
    let mut arr: [f32; 1] = [2.0];
    arr[0] -= 1.0;
    d -= arr[0];

    // Field projection:
    let mut foo: Foo<f32> = Foo(2.0);
    let mut bar: Bar<f32> = Bar { x: 2.0 };
    foo.0 -= 1.0;
    bar.x -= 1.0;
    d -= foo.0 + bar.x;

    // Array + Field projection:
    let mut arr: [Foo<f32>; 1] = [Foo(2.0)];
    arr[0].0 -= 1.0;
    d -= arr[0].0;
    let mut arr: [Bar<f32>; 1] = [Bar { x: 2.0 }];
    arr[0].x -= 1.0;
    d -= arr[0].x;

    // Field + Array projection:
    let mut arr: Foo<[f32; 1]> = Foo([2.0]);
    (arr.0)[0] -= 1.0;
    d -= (arr.0)[0];
    let mut arr: Bar<[f32; 1]> = Bar { x: [2.0] };
    arr.x[0] -= 1.0;
    d -= arr.x[0];

    d
}

macro_rules! test {
    ($c:ident, $e:expr, $r:expr) => {
        const $c: f32 = $e;
        assert_eq!($c, $r);
        assert_eq!($e, $r);
    }
}

fn main() {
    test!(BASICS, basics((2.0,)), 13.0);
    test!(ADD, add_assign(W(1.0)), 10.0);
    test!(MUL, mul_assign(A { a: 0.0 }), 256.0);
    test!(DIV, div_assign([1.0]), 4.0);
    test!(REM, rem_assign(W(5.0)), 5.0);
    test!(SUB, sub_assign(W(8.0)), 0.0);
}
