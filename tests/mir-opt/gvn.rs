//@ test-mir-pass: GVN
//@ compile-flags: -Zdump-mir-exclude-alloc-bytes
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ only-64bit

#![feature(rustc_attrs)]
#![feature(custom_mir)]
#![feature(core_intrinsics)]
#![feature(freeze)]
#![allow(ambiguous_wide_pointer_comparisons)]
#![allow(unconditional_panic)]
#![allow(unused)]

use std::intrinsics::mir::*;
use std::marker::{Freeze, PhantomData};
use std::mem::transmute;

struct S<T>(T);

fn subexpression_elimination(x: u64, y: u64, mut z: u64) {
    // CHECK-LABEL: fn subexpression_elimination(

    // CHECK: [[add:_.*]] = Add(copy _1, copy _2);
    // CHECK: opaque::<u64>(copy [[add]])
    opaque(x + y);
    // CHECK: [[mul:_.*]] = Mul(copy _1, copy _2);
    // CHECK: opaque::<u64>(copy [[mul]])
    opaque(x * y);
    // CHECK: [[sub:_.*]] = Sub(copy _1, copy _2);
    // CHECK: opaque::<u64>(copy [[sub]])
    opaque(x - y);
    // CHECK: [[div:_.*]] = Div(copy _1, copy _2);
    // CHECK: opaque::<u64>(copy [[div]])
    opaque(x / y);
    // CHECK: [[rem:_.*]] = Rem(copy _1, copy _2);
    // CHECK: opaque::<u64>(copy [[rem]])
    opaque(x % y);
    // CHECK: [[and:_.*]] = BitAnd(copy _1, copy _2);
    // CHECK: opaque::<u64>(copy [[and]])
    opaque(x & y);
    // CHECK: [[or:_.*]] = BitOr(copy _1, copy _2);
    // CHECK: opaque::<u64>(copy [[or]])
    opaque(x | y);
    // CHECK: [[xor:_.*]] = BitXor(copy _1, copy _2);
    // CHECK: opaque::<u64>(copy [[xor]])
    opaque(x ^ y);
    // CHECK: [[shl:_.*]] = Shl(copy _1, copy _2);
    // CHECK: opaque::<u64>(copy [[shl]])
    opaque(x << y);
    // CHECK: [[shr:_.*]] = Shr(copy _1, copy _2);
    // CHECK: opaque::<u64>(copy [[shr]])
    opaque(x >> y);
    // CHECK: [[int:_.*]] = copy _1 as u32 (IntToInt);
    // CHECK: opaque::<u32>(copy [[int]])
    opaque(x as u32);
    // CHECK: [[float:_.*]] = copy _1 as f32 (IntToFloat);
    // CHECK: opaque::<f32>(copy [[float]])
    opaque(x as f32);
    // CHECK: [[wrap:_.*]] = S::<u64>(copy _1);
    // CHECK: opaque::<S<u64>>(copy [[wrap]])
    opaque(S(x));
    // CHECK: opaque::<u64>(copy _1)
    opaque(S(x).0);

    // Those are duplicates to substitute somehow.
    // CHECK: opaque::<u64>(copy [[add]])
    opaque(x + y);
    // CHECK: opaque::<u64>(copy [[mul]])
    opaque(x * y);
    // CHECK: opaque::<u64>(copy [[sub]])
    opaque(x - y);
    // CHECK: opaque::<u64>(copy [[div]])
    opaque(x / y);
    // CHECK: opaque::<u64>(copy [[rem]])
    opaque(x % y);
    // CHECK: opaque::<u64>(copy [[and]])
    opaque(x & y);
    // CHECK: opaque::<u64>(copy [[or]])
    opaque(x | y);
    // CHECK: opaque::<u64>(copy [[xor]])
    opaque(x ^ y);
    // CHECK: opaque::<u64>(copy [[shl]])
    opaque(x << y);
    // CHECK: opaque::<u64>(copy [[shr]])
    opaque(x >> y);
    // CHECK: opaque::<u32>(copy [[int]])
    opaque(x as u32);
    // CHECK: opaque::<f32>(copy [[float]])
    opaque(x as f32);
    // CHECK: opaque::<S<u64>>(copy [[wrap]])
    opaque(S(x));
    // CHECK: opaque::<u64>(copy _1)
    opaque(S(x).0);

    // We can substitute through a complex expression.
    // CHECK: [[compound:_.*]] = Sub(copy [[mul]], copy _2);
    // CHECK: opaque::<u64>(copy [[compound]])
    // CHECK: opaque::<u64>(copy [[compound]])
    opaque((x * y) - y);
    opaque((x * y) - y);

    // We cannot substitute through an immutable reference.
    // CHECK: [[ref:_.*]] = &_3;
    // CHECK: [[deref:_.*]] = copy (*[[ref]]);
    // CHECK: [[addref:_.*]] = Add(move [[deref]], copy _1);
    // CHECK: opaque::<u64>(move [[addref]])
    // CHECK: [[deref2:_.*]] = copy (*[[ref]]);
    // CHECK: [[addref2:_.*]] = Add(move [[deref2]], copy _1);
    // CHECK: opaque::<u64>(move [[addref2]])
    let a = &z;
    opaque(*a + x);
    opaque(*a + x);

    // But not through a mutable reference or a pointer.
    // CHECK: [[mut:_.*]] = &mut _3;
    // CHECK: [[addmut:_.*]] = Add(
    // CHECK: opaque::<u64>(move [[addmut]])
    // CHECK: [[addmut2:_.*]] = Add(
    // CHECK: opaque::<u64>(move [[addmut2]])
    let b = &mut z;
    opaque(*b + x);
    opaque(*b + x);
    unsafe {
        // CHECK: [[raw:_.*]] = &raw const _3;
        // CHECK: [[addraw:_.*]] = Add(
        // CHECK: opaque::<u64>(move [[addraw]])
        // CHECK: [[addraw2:_.*]] = Add(
        // CHECK: opaque::<u64>(move [[addraw2]])
        let c = &raw const z;
        opaque(*c + x);
        opaque(*c + x);
        // CHECK: [[ptr:_.*]] = &raw mut _3;
        // CHECK: [[addptr:_.*]] = Add(
        // CHECK: opaque::<u64>(move [[addptr]])
        // CHECK: [[addptr2:_.*]] = Add(
        // CHECK: opaque::<u64>(move [[addptr2]])
        let d = &raw mut z;
        opaque(*d + x);
        opaque(*d + x);
    }

    // We still cannot substitute again, and never with the earlier computations.
    // Important: `e` is not `a`!
    // CHECK: [[ref2:_.*]] = &_3;
    // CHECK: [[deref2:_.*]] = copy (*[[ref2]]);
    // CHECK: [[addref2:_.*]] = Add(move [[deref2]], copy _1);
    // CHECK: opaque::<u64>(move [[addref2]])
    // CHECK: [[deref3:_.*]] = copy (*[[ref2]]);
    // CHECK: [[addref3:_.*]] = Add(move [[deref3]], copy _1);
    // CHECK: opaque::<u64>(move [[addref3]])
    let e = &z;
    opaque(*e + x);
    opaque(*e + x);
}

fn wrap_unwrap<T: Copy>(x: T) -> T {
    // CHECK-LABEL: fn wrap_unwrap(
    // CHECK: [[some:_.*]] = Option::<T>::Some(copy _1);
    // CHECK: switchInt(const 1_isize)
    // CHECK: _0 = copy _1;
    match Some(x) {
        Some(y) => y,
        None => panic!(),
    }
}

fn repeated_index<T: Copy, const N: usize>(x: T, idx: usize) {
    // CHECK-LABEL: fn repeated_index(
    // CHECK: [[a:_.*]] = [copy _1; N];
    let a = [x; N];
    // CHECK: opaque::<T>(copy _1)
    opaque(a[0]);
    // CHECK: opaque::<T>(copy _1)
    opaque(a[idx]);
}

fn unary(x: i64) {
    // CHECK-LABEL: fn unary(
    // CHECK: opaque::<i64>(copy _1)
    opaque(--x); // This is `x`.

    // CHECK: [[b:_.*]] = Lt(copy _1, const 13_i64);
    // CHECK: opaque::<bool>(copy [[b]])
    let b = x < 13;
    opaque(!!b); // This is `b`.

    // Both lines should test the same thing.
    // CHECK: [[c:_.*]] = Ne(copy _1, const 15_i64);
    // CHECK: opaque::<bool>(copy [[c]])
    // CHECK: opaque::<bool>(copy [[c]])
    opaque(x != 15);
    opaque(!(x == 15));

    // Both lines should test the same thing.
    // CHECK: [[d:_.*]] = Eq(copy _1, const 35_i64);
    // CHECK: opaque::<bool>(copy [[d]])
    // CHECK: opaque::<bool>(copy [[d]])
    opaque(x == 35);
    opaque(!(x != 35));
}

/// Verify symbolic integer arithmetic simplifications.
fn arithmetic(x: u64) {
    // CHECK-LABEL: fn arithmetic(
    // CHECK: opaque::<u64>(copy _1)
    opaque(x + 0);
    // CHECK: opaque::<u64>(copy _1)
    opaque(x - 0);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x - x);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x * 0);
    // CHECK: opaque::<u64>(copy _1)
    opaque(x * 1);
    // CHECK: assert(!const true, "attempt to divide `{}` by zero",
    // CHECK: [[div0:_.*]] = Div(copy _1, const 0_u64);
    // CHECK: opaque::<u64>(move [[div0]])
    opaque(x / 0);
    // CHECK: opaque::<u64>(copy _1)
    opaque(x / 1);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(0 / x);
    // CHECK: [[odiv:_.*]] = Div(const 1_u64, copy _1);
    // CHECK: opaque::<u64>(move [[odiv]])
    opaque(1 / x);
    // CHECK: assert(!const true, "attempt to calculate the remainder of `{}` with a divisor of zero"
    // CHECK: [[rem0:_.*]] = Rem(copy _1, const 0_u64);
    // CHECK: opaque::<u64>(move [[rem0]])
    opaque(x % 0);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x % 1);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(0 % x);
    // CHECK: [[orem:_.*]] = Rem(const 1_u64, copy _1);
    // CHECK: opaque::<u64>(move [[orem]])
    opaque(1 % x);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x & 0);
    // CHECK: opaque::<u64>(copy _1)
    opaque(x & u64::MAX);
    // CHECK: opaque::<u64>(copy _1)
    opaque(x | 0);
    // CHECK: opaque::<u64>(const u64::MAX)
    opaque(x | u64::MAX);
    // CHECK: opaque::<u64>(copy _1)
    opaque(x ^ 0);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x ^ x);
    // CHECK: opaque::<u64>(copy _1)
    opaque(x >> 0);
    // CHECK: opaque::<u64>(copy _1)
    opaque(x << 0);
}

fn comparison(x: u64, y: u64) {
    // CHECK-LABEL: fn comparison(
    // CHECK: opaque::<bool>(const true)
    opaque(x == x);
    // CHECK: opaque::<bool>(const false)
    opaque(x != x);
    // CHECK: [[eqxy:_.*]] = Eq(copy _1, copy _2);
    // CHECK: opaque::<bool>(move [[eqxy]])
    opaque(x == y);
    // CHECK: [[nexy:_.*]] = Ne(copy _1, copy _2);
    // CHECK: opaque::<bool>(move [[nexy]])
    opaque(x != y);
}

/// Verify symbolic integer arithmetic simplifications on checked ops.
#[rustc_inherit_overflow_checks]
fn arithmetic_checked(x: u64) {
    // CHECK-LABEL: fn arithmetic_checked(
    // CHECK: assert(!const false,
    // CHECK: opaque::<u64>(copy _1)
    opaque(x + 0);
    // CHECK: assert(!const false,
    // CHECK: opaque::<u64>(copy _1)
    opaque(x - 0);
    // CHECK: assert(!const false,
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x - x);
    // CHECK: assert(!const false,
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x * 0);
    // CHECK: assert(!const false,
    // CHECK: opaque::<u64>(copy _1)
    opaque(x * 1);
}

/// Verify that we do not apply arithmetic simplifications on floats.
fn arithmetic_float(x: f64) {
    // CHECK-LABEL: fn arithmetic_float(
    // CHECK: [[add:_.*]] = Add(copy _1, const 0f64);
    // CHECK: opaque::<f64>(move [[add]])
    opaque(x + 0.);
    // CHECK: [[sub:_.*]] = Sub(copy _1, const 0f64);
    // CHECK: opaque::<f64>(move [[sub]])
    opaque(x - 0.);
    // CHECK: [[mul:_.*]] = Mul(copy _1, const 0f64);
    // CHECK: opaque::<f64>(move [[mul]])
    opaque(x * 0.);
    // CHECK: [[div0:_.*]] = Div(copy _1, const 0f64);
    // CHECK: opaque::<f64>(move [[div0]])
    opaque(x / 0.);
    // CHECK: [[zdiv:_.*]] = Div(const 0f64, copy _1);
    // CHECK: opaque::<f64>(move [[zdiv]])
    opaque(0. / x);
    // CHECK: [[rem0:_.*]] = Rem(copy _1, const 0f64);
    // CHECK: opaque::<f64>(move [[rem0]])
    opaque(x % 0.);
    // CHECK: [[zrem:_.*]] = Rem(const 0f64, copy _1);
    // CHECK: opaque::<f64>(move [[zrem]])
    opaque(0. % x);
    // Those are not simplifiable to `true`/`false`, thanks to NaNs.
    // CHECK: [[eq:_.*]] = Eq(copy _1, copy _1);
    // CHECK: opaque::<bool>(move [[eq]])
    opaque(x == x);
    // CHECK: [[ne:_.*]] = Ne(copy _1, copy _1);
    // CHECK: opaque::<bool>(move [[ne]])
    opaque(x != x);
}

fn cast() {
    // CHECK-LABEL: fn cast(
    let i = 1_i64;
    let u = 1_u64;
    let f = 1_f64;
    // CHECK: opaque::<u8>(const 1_u8)
    opaque(i as u8);
    // CHECK: opaque::<u16>(const 1_u16)
    opaque(i as u16);
    // CHECK: opaque::<u32>(const 1_u32)
    opaque(i as u32);
    // CHECK: opaque::<u64>(const 1_u64)
    opaque(i as u64);
    // CHECK: opaque::<i8>(const 1_i8)
    opaque(i as i8);
    // CHECK: opaque::<i16>(const 1_i16)
    opaque(i as i16);
    // CHECK: opaque::<i32>(const 1_i32)
    opaque(i as i32);
    // CHECK: opaque::<i64>(const 1_i64)
    opaque(i as i64);
    // CHECK: opaque::<f32>(const 1f32)
    opaque(i as f32);
    // CHECK: opaque::<f64>(const 1f64)
    opaque(i as f64);
    // CHECK: opaque::<u8>(const 1_u8)
    opaque(u as u8);
    // CHECK: opaque::<u16>(const 1_u16)
    opaque(u as u16);
    // CHECK: opaque::<u32>(const 1_u32)
    opaque(u as u32);
    // CHECK: opaque::<u64>(const 1_u64)
    opaque(u as u64);
    // CHECK: opaque::<i8>(const 1_i8)
    opaque(u as i8);
    // CHECK: opaque::<i16>(const 1_i16)
    opaque(u as i16);
    // CHECK: opaque::<i32>(const 1_i32)
    opaque(u as i32);
    // CHECK: opaque::<i64>(const 1_i64)
    opaque(u as i64);
    // CHECK: opaque::<f32>(const 1f32)
    opaque(u as f32);
    // CHECK: opaque::<f64>(const 1f64)
    opaque(u as f64);
    // CHECK: opaque::<u8>(const 1_u8)
    opaque(f as u8);
    // CHECK: opaque::<u16>(const 1_u16)
    opaque(f as u16);
    // CHECK: opaque::<u32>(const 1_u32)
    opaque(f as u32);
    // CHECK: opaque::<u64>(const 1_u64)
    opaque(f as u64);
    // CHECK: opaque::<i8>(const 1_i8)
    opaque(f as i8);
    // CHECK: opaque::<i16>(const 1_i16)
    opaque(f as i16);
    // CHECK: opaque::<i32>(const 1_i32)
    opaque(f as i32);
    // CHECK: opaque::<i64>(const 1_i64)
    opaque(f as i64);
    // CHECK: opaque::<f32>(const 1f32)
    opaque(f as f32);
    // CHECK: opaque::<f64>(const 1f64)
    opaque(f as f64);
}

fn multiple_branches(t: bool, x: u8, y: u8) {
    // CHECK-LABEL: fn multiple_branches(
    // CHECK: switchInt(copy _1) -> [0: [[bbf:bb.*]], otherwise: [[bbt:bb.*]]];
    if t {
        // CHECK: [[bbt]]: {
        // CHECK: [[a:_.*]] = Add(copy _2, copy _3);
        // CHECK: opaque::<u8>(copy [[a]])
        // CHECK: opaque::<u8>(copy [[a]])
        // CHECK: goto -> [[bbc:bb.*]];
        opaque(x + y);
        opaque(x + y);
    } else {
        // CHECK: [[bbf]]: {
        // CHECK: [[b:_.*]] = Add(copy _2, copy _3);
        // CHECK: opaque::<u8>(copy [[b]])
        // CHECK: opaque::<u8>(copy [[b]])
        // CHECK: goto -> [[bbc:bb.*]];
        opaque(x + y);
        opaque(x + y);
    }
    // Neither `a` nor `b` dominate `c`, so we cannot reuse any of them.
    // CHECK: [[bbc]]: {
    // CHECK: [[c:_.*]] = Add(copy _2, copy _3);
    // CHECK: opaque::<u8>(copy [[c]])
    opaque(x + y);

    // `c` dominates both calls, so we can reuse it.
    if t {
        // CHECK: opaque::<u8>(copy [[c]])
        opaque(x + y);
    } else {
        // CHECK: opaque::<u8>(copy [[c]])
        opaque(x + y);
    }
}

/// Verify that we do not reuse a `&raw? mut?` rvalue.
fn references(mut x: impl Sized) {
    // CHECK-LABEL: fn references(
    // CHECK: [[ref1:_.*]] = &_1;
    // CHECK: opaque::<&impl Sized>(move [[ref1]])
    opaque(&x);
    // CHECK: [[ref2:_.*]] = &_1;
    // CHECK: opaque::<&impl Sized>(move [[ref2]])
    opaque(&x);
    // CHECK: [[ref3:_.*]] = &mut _1;
    // CHECK: opaque::<&mut impl Sized>(move [[ref3]])
    opaque(&mut x);
    // CHECK: [[ref4:_.*]] = &mut _1;
    // CHECK: opaque::<&mut impl Sized>(move [[ref4]])
    opaque(&mut x);
    // CHECK: [[ref5:_.*]] = &raw const _1;
    // CHECK: opaque::<*const impl Sized>(move [[ref5]])
    opaque(&raw const x);
    // CHECK: [[ref6:_.*]] = &raw const _1;
    // CHECK: opaque::<*const impl Sized>(move [[ref6]])
    opaque(&raw const x);
    // CHECK: [[ref7:_.*]] = &raw mut _1;
    // CHECK: opaque::<*mut impl Sized>(move [[ref7]])
    opaque(&raw mut x);
    // CHECK: [[ref8:_.*]] = &raw mut _1;
    // CHECK: opaque::<*mut impl Sized>(move [[ref8]])
    opaque(&raw mut x);

    let r = &mut x;
    let s = S(r).0; // Obfuscate `r`. Following lines should still reborrow `r`.
    // CHECK: [[ref9:_.*]] = &mut _1;
    // CHECK: [[ref10:_.*]] = &(*[[ref9]]);
    // CHECK: opaque::<&impl Sized>(move [[ref10]])
    opaque(&*s);
    // CHECK: [[ref11:_.*]] = &mut (*[[ref9]]);
    // CHECK: opaque::<&mut impl Sized>(move [[ref11]])
    opaque(&mut *s);
    // CHECK: [[ref12:_.*]] = &raw const (*[[ref9]]);
    // CHECK: opaque::<*const impl Sized>(move [[ref12]])
    opaque(&raw const *s);
    // CHECK: [[ref12:_.*]] = &raw mut (*[[ref9]]);
    // CHECK: opaque::<*mut impl Sized>(move [[ref12]])
    opaque(&raw mut *s);
}

fn dereferences(t: &mut u32, u: &impl Copy, s: &S<u32>) {
    // CHECK-LABEL: fn dereferences(

    // Do not reuse dereferences of `&mut`.
    // CHECK: [[st1:_.*]] = copy (*_1);
    // CHECK: opaque::<u32>(move [[st1]])
    // CHECK: [[st2:_.*]] = copy (*_1);
    // CHECK: opaque::<u32>(move [[st2]])
    opaque(*t);
    opaque(*t);

    // Do not reuse dereferences of `*const`.
    // CHECK: [[raw:_.*]] = &raw const (*_1);
    // CHECK: [[st3:_.*]] = copy (*[[raw]]);
    // CHECK: opaque::<u32>(move [[st3]])
    // CHECK: [[st4:_.*]] = copy (*[[raw]]);
    // CHECK: opaque::<u32>(move [[st4]])
    let z = &raw const *t;
    unsafe { opaque(*z) };
    unsafe { opaque(*z) };

    // Do not reuse dereferences of `*mut`.
    // CHECK: [[ptr:_.*]] = &raw mut (*_1);
    // CHECK: [[st5:_.*]] = copy (*[[ptr]]);
    // CHECK: opaque::<u32>(move [[st5]])
    // CHECK: [[st6:_.*]] = copy (*[[ptr]]);
    // CHECK: opaque::<u32>(move [[st6]])
    let z = &raw mut *t;
    unsafe { opaque(*z) };
    unsafe { opaque(*z) };

    // Do not reuse dereferences of `&Freeze`.
    // CHECK: [[ref:_.*]] = &(*_1);
    // CHECK: [[st7:_.*]] = copy (*[[ref]]);
    // CHECK: opaque::<u32>(move [[st7]])
    // CHECK: [[st8:_.*]] = copy (*[[ref]]);
    // CHECK: opaque::<u32>(move [[st8]])
    let z = &*t;
    opaque(*z);
    opaque(*z);
    // Not in reborrows either.
    // CHECK: [[reborrow:_.*]] = &(*[[ref]]);
    // CHECK: opaque::<&u32>(move [[reborrow]])
    opaque(&*z);

    // `*u` is not Freeze, so we cannot reuse.
    // CHECK: [[st8:_.*]] = copy (*_2);
    // CHECK: opaque::<impl Copy>(move [[st8]])
    // CHECK: [[st9:_.*]] = copy (*_2);
    // CHECK: opaque::<impl Copy>(move [[st9]])
    opaque(*u);
    opaque(*u);

    // `*s` is not Copy, but `(*s).0` is, but we still cannot reuse.
    // CHECK: [[st10:_.*]] = copy ((*_3).0: u32);
    // CHECK: opaque::<u32>(move [[st10]])
    // CHECK: [[st11:_.*]] = copy ((*_3).0: u32);
    // CHECK: opaque::<u32>(move [[st11]])
    opaque(s.0);
    opaque(s.0);
}

fn slices() {
    // CHECK-LABEL: fn slices(
    // CHECK: {{_.*}} = const "
    // CHECK-NOT: {{_.*}} = const "
    let s = "my favourite slice"; // This is a `Const::Slice` in MIR.
    opaque(s);
    let t = s; // This should be the same pointer, so cannot be a `Const::Slice`.
    opaque(t);
    assert_eq!(s.as_ptr(), t.as_ptr());
    let u = unsafe { transmute::<&str, &[u8]>(s) };
    opaque(u);
    assert_eq!(s.as_ptr(), u.as_ptr());
}

#[custom_mir(dialect = "analysis")]
fn duplicate_slice() -> (bool, bool) {
    // CHECK-LABEL: fn duplicate_slice(
    mir! {
        let au: u128;
        let bu: u128;
        let cu: u128;
        let du: u128;
        let c: &str;
        let d: &str;
        {
            // CHECK: [[a:_.*]] = (const "a",);
            // CHECK: [[au:_.*]] = copy ([[a]].0: &str) as u128 (Transmute);
            let a = ("a",);
            Call(au = transmute::<_, u128>(a.0), ReturnTo(bb1), UnwindContinue())
        }
        bb1 = {
            // CHECK: [[c:_.*]] = identity::<&str>(copy ([[a]].0: &str))
            Call(c = identity(a.0), ReturnTo(bb2), UnwindContinue())
        }
        bb2 = {
            // CHECK: [[cu:_.*]] = copy [[c]] as u128 (Transmute);
            Call(cu = transmute::<_, u128>(c), ReturnTo(bb3), UnwindContinue())
        }
        bb3 = {
            // This slice is different from `a.0`. Hence `bu` is not `au`.
            // CHECK: [[b:_.*]] = const "a";
            // CHECK: [[bu:_.*]] = copy [[b]] as u128 (Transmute);
            let b = "a";
            Call(bu = transmute::<_, u128>(b), ReturnTo(bb4), UnwindContinue())
        }
        bb4 = {
            // This returns a copy of `b`, which is not `a`.
            // CHECK: [[d:_.*]] = identity::<&str>(copy [[b]])
            Call(d = identity(b), ReturnTo(bb5), UnwindContinue())
        }
        bb5 = {
            // CHECK: [[du:_.*]] = copy [[d]] as u128 (Transmute);
            Call(du = transmute::<_, u128>(d), ReturnTo(bb6), UnwindContinue())
        }
        bb6 = {
            // `direct` must not fold to `true`, as `indirect` will not.
            // CHECK: = Eq(copy [[au]], copy [[bu]]);
            // CHECK: = Eq(copy [[cu]], copy [[du]]);
            let direct = au == bu;
            let indirect = cu == du;
            RET = (direct, indirect);
            Return()
        }
    }
}

fn repeat() {
    // CHECK-LABEL: fn repeat(
    // CHECK: = [const 5_i32; 10];
    let val = 5;
    let array = [val, val, val, val, val, val, val, val, val, val];
}

/// Verify that we do not merge fn pointers created by casts.
fn fn_pointers() {
    // CHECK-LABEL: fn fn_pointers(
    // CHECK: [[f:_.*]] = identity::<u8> as fn(u8) -> u8 (PointerCoercion(ReifyFnPointer
    // CHECK: opaque::<fn(u8) -> u8>(copy [[f]])
    let f = identity as fn(u8) -> u8;
    opaque(f);
    // CHECK: [[g:_.*]] = identity::<u8> as fn(u8) -> u8 (PointerCoercion(ReifyFnPointer
    // CHECK: opaque::<fn(u8) -> u8>(copy [[g]])
    let g = identity as fn(u8) -> u8;
    opaque(g);

    // CHECK: [[cf:_.*]] = const {{.*}} as fn() (PointerCoercion(ClosureFnPointer
    // CHECK: opaque::<fn()>(copy [[cf]])
    let closure = || {};
    let cf = closure as fn();
    opaque(cf);
    // CHECK: [[cg:_.*]] = const {{.*}} as fn() (PointerCoercion(ClosureFnPointer
    // CHECK: opaque::<fn()>(copy [[cg]])
    let cg = closure as fn();
    opaque(cg);
}

/// Verify that we do not create a `ConstValue::Indirect` backed by a static's AllocId.
#[custom_mir(dialect = "analysis")]
fn indirect_static() {
    static A: Option<u8> = None;

    mir! {
        {
            let ptr = Static(A);
            let out = Field::<u8>(Variant(*ptr, 1), 0);
            Return()
        }
    }
}

/// Verify that having constant index `u64::MAX` does not yield to an overflow in rustc.
fn constant_index_overflow<T: Copy>(x: &[T]) {
    // CHECK-LABEL: fn constant_index_overflow(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];
    // CHECK: [[a]] = const usize::MAX;
    // CHECK-NOT: = (*_1)[{{.*}} of 0];
    // CHECK: [[b]] = copy (*_1)[[[a]]];
    // CHECK-NOT: = (*_1)[{{.*}} of 0];
    // CHECK: [[b]] = copy (*_1)[0 of 1];
    // CHECK-NOT: = (*_1)[{{.*}} of 0];
    let a = u64::MAX as usize;
    let b = if a < x.len() { x[a] } else { x[0] };
    opaque(b)
}

/// Check that we do not attempt to simplify anything when there is provenance.
fn wide_ptr_provenance() {
    // CHECK-LABEL: fn wide_ptr_provenance(
    let a: *const dyn Send = &1 as &dyn Send;
    let b: *const dyn Send = &1 as &dyn Send;

    // CHECK: [[eqp:_.*]] = Eq(copy [[a:_.*]], copy [[b:_.*]]);
    // CHECK: opaque::<bool>(move [[eqp]])
    opaque(a == b);
    // CHECK: [[nep:_.*]] = Ne(copy [[a]], copy [[b]]);
    // CHECK: opaque::<bool>(move [[nep]])
    opaque(a != b);
    // CHECK: [[ltp:_.*]] = Lt(copy [[a]], copy [[b]]);
    // CHECK: opaque::<bool>(move [[ltp]])
    opaque(a < b);
    // CHECK: [[lep:_.*]] = Le(copy [[a]], copy [[b]]);
    // CHECK: opaque::<bool>(move [[lep]])
    opaque(a <= b);
    // CHECK: [[gtp:_.*]] = Gt(copy [[a]], copy [[b]]);
    // CHECK: opaque::<bool>(move [[gtp]])
    opaque(a > b);
    // CHECK: [[gep:_.*]] = Ge(copy [[a]], copy [[b]]);
    // CHECK: opaque::<bool>(move [[gep]])
    opaque(a >= b);
}

/// Both pointers come form the same allocation, so we could probably fold the comparisons.
fn wide_ptr_same_provenance() {
    // CHECK-LABEL: fn wide_ptr_same_provenance(
    let slice = &[1, 2];
    let a: *const dyn Send = &slice[0] as &dyn Send;
    let b: *const dyn Send = &slice[1] as &dyn Send;

    // CHECK: [[eqp:_.*]] = Eq(copy [[a:_.*]], copy [[b:_.*]]);
    // CHECK: opaque::<bool>(move [[eqp]])
    opaque(a == b);
    // CHECK: [[nep:_.*]] = Ne(copy [[a]], copy [[b]]);
    // CHECK: opaque::<bool>(move [[nep]])
    opaque(a != b);
    // CHECK: [[ltp:_.*]] = Lt(copy [[a]], copy [[b]]);
    // CHECK: opaque::<bool>(move [[ltp]])
    opaque(a < b);
    // CHECK: [[lep:_.*]] = Le(copy [[a]], copy [[b]]);
    // CHECK: opaque::<bool>(move [[lep]])
    opaque(a <= b);
    // CHECK: [[gtp:_.*]] = Gt(copy [[a]], copy [[b]]);
    // CHECK: opaque::<bool>(move [[gtp]])
    opaque(a > b);
    // CHECK: [[gep:_.*]] = Ge(copy [[a]], copy [[b]]);
    // CHECK: opaque::<bool>(move [[gep]])
    opaque(a >= b);
}

/// Check that we do simplify when there is no provenance, and do not ICE.
fn wide_ptr_integer() {
    // CHECK-LABEL: fn wide_ptr_integer(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];

    let a: *const [u8] = unsafe { transmute((1usize, 1usize)) };
    let b: *const [u8] = unsafe { transmute((1usize, 2usize)) };

    // CHECK: opaque::<bool>(const false)
    opaque(a == b);
    // CHECK: opaque::<bool>(const true)
    opaque(a != b);
    // CHECK: opaque::<bool>(const true)
    opaque(a < b);
    // CHECK: opaque::<bool>(const true)
    opaque(a <= b);
    // CHECK: opaque::<bool>(const false)
    opaque(a > b);
    // CHECK: opaque::<bool>(const false)
    opaque(a >= b);
}

#[custom_mir(dialect = "analysis", phase = "post-cleanup")]
fn borrowed<T: Copy + Freeze>(x: T) {
    // CHECK-LABEL: fn borrowed(
    // CHECK: bb0: {
    // CHECK-NEXT: _2 = copy _1;
    // CHECK-NEXT: _3 = &_1;
    // CHECK-NEXT: _0 = opaque::<&T>(copy _3)
    // CHECK: bb1: {
    // CHECK-NEXT: _0 = opaque::<T>(copy _1)
    // CHECK: bb2: {
    // CHECK-NEXT: _0 = opaque::<T>(copy _1)
    mir! {
        {
            let a = x;
            let r1 = &x;
            Call(RET = opaque(r1), ReturnTo(next), UnwindContinue())
        }
        next = {
            Call(RET = opaque(a), ReturnTo(deref), UnwindContinue())
        }
        deref = {
            Call(RET = opaque(*r1), ReturnTo(ret), UnwindContinue())
        }
        ret = {
            Return()
        }
    }
}

/// Generic type `T` is not known to be `Freeze`, so shared borrows may be mutable.
#[custom_mir(dialect = "analysis", phase = "post-cleanup")]
fn non_freeze<T: Copy>(x: T) {
    // CHECK-LABEL: fn non_freeze(
    // CHECK: bb0: {
    // CHECK-NEXT: _2 = copy _1;
    // CHECK-NEXT: _3 = &_1;
    // CHECK-NEXT: _0 = opaque::<&T>(copy _3)
    // CHECK: bb1: {
    // CHECK-NEXT: _0 = opaque::<T>(copy _2)
    // CHECK: bb2: {
    // CHECK-NEXT: _0 = opaque::<T>(copy (*_3))
    mir! {
        {
            let a = x;
            let r1 = &x;
            Call(RET = opaque(r1), ReturnTo(next), UnwindContinue())
        }
        next = {
            Call(RET = opaque(a), ReturnTo(deref), UnwindContinue())
        }
        deref = {
            Call(RET = opaque(*r1), ReturnTo(ret), UnwindContinue())
        }
        ret = {
            Return()
        }
    }
}

// Check that we can const-prop into `from_raw_parts`
fn slice_const_length(x: &[i32]) -> *const [i32] {
    // CHECK-LABEL: fn slice_const_length(
    // CHECK: _0 = *const [i32] from (copy {{_[0-9]+}}, const 123_usize);
    let ptr = x.as_ptr();
    let len = 123;
    std::intrinsics::aggregate_raw_ptr(ptr, len)
}

fn meta_of_ref_to_slice(x: *const i32) -> usize {
    // CHECK-LABEL: fn meta_of_ref_to_slice
    // CHECK: _0 = const 1_usize
    let ptr: *const [i32] = std::intrinsics::aggregate_raw_ptr(x, 1);
    std::intrinsics::ptr_metadata(ptr)
}

fn slice_from_raw_parts_as_ptr(x: *const u16, n: usize) -> (*const u16, *const f32) {
    // CHECK-LABEL: fn slice_from_raw_parts_as_ptr
    // CHECK: _8 = copy _1 as *const f32 (PtrToPtr);
    // CHECK: _0 = (copy _1, move _8);
    let ptr: *const [u16] = std::intrinsics::aggregate_raw_ptr(x, n);
    (ptr as *const u16, ptr as *const f32)
}

fn casts_before_aggregate_raw_ptr(x: *const u32) -> *const [u8] {
    // CHECK-LABEL: fn casts_before_aggregate_raw_ptr
    // CHECK: _0 = *const [u8] from (copy _1, const 4_usize);
    let x = x as *const [u8; 4];
    let x = x as *const u8;
    let x = x as *const ();
    std::intrinsics::aggregate_raw_ptr(x, 4)
}

fn manual_slice_mut_len(x: &mut [i32]) -> usize {
    // CHECK-LABEL: fn manual_slice_mut_len
    // CHECK: _0 = PtrMetadata(copy _1);
    let x: *mut [i32] = x;
    let x: *const [i32] = x;
    std::intrinsics::ptr_metadata(x)
}

// `.len()` on arrays ends up being something like this
fn array_len(x: &mut [i32; 42]) -> usize {
    // CHECK-LABEL: fn array_len
    // CHECK: _0 = const 42_usize;
    let x: &[i32] = x;
    std::intrinsics::ptr_metadata(x)
}

// Check that we only load the length once, rather than all 3 times.
fn dedup_multiple_bounds_checks_lengths(x: &[i32]) -> [i32; 3] {
    // CHECK-LABEL: fn dedup_multiple_bounds_checks_lengths
    // CHECK: [[LEN:_.+]] = PtrMetadata(copy _1);
    // CHECK: Lt(const 42_usize, copy [[LEN]]);
    // CHECK: assert{{.+}}copy [[LEN]]
    // CHECK: [[A:_.+]] = copy (*_1)[42 of 43];
    // CHECK-NOT: PtrMetadata
    // CHECK: Lt(const 13_usize, copy [[LEN]]);
    // CHECK: assert{{.+}}copy [[LEN]]
    // CHECK: [[B:_.+]] = copy (*_1)[13 of 14];
    // CHECK-NOT: PtrMetadata
    // CHECK: Lt(const 7_usize, copy [[LEN]]);
    // CHECK: assert{{.+}}copy [[LEN]]
    // CHECK: [[C:_.+]] = copy (*_1)[7 of 8];
    // CHECK: _0 = [move [[A]], move [[B]], move [[C]]]
    [x[42], x[13], x[7]]
}

#[custom_mir(dialect = "runtime")]
fn generic_cast_metadata<T, A: ?Sized, B: ?Sized>(ps: *const [T], pa: *const A, pb: *const B) {
    // CHECK-LABEL: fn generic_cast_metadata
    mir! {
        {
            // These tests check that we correctly do or don't elide casts
            // when the pointee metadata do or don't match, respectively.

            // Metadata usize -> (), do not optimize.
            // CHECK: [[T:_.+]] = copy _1 as
            // CHECK-NEXT: PtrMetadata(copy [[T]])
            let t1 = CastPtrToPtr::<_, *const T>(ps);
            let m1 = PtrMetadata(t1);

            // `(&A, [T])` has `usize` metadata, same as `[T]`, yes optimize.
            // CHECK: [[T:_.+]] = copy _1 as
            // CHECK-NEXT: PtrMetadata(copy _1)
            let t2 = CastPtrToPtr::<_, *const (&A, [T])>(ps);
            let m2 = PtrMetadata(t2);

            // Tail `A` and tail `B`, do not optimize.
            // CHECK: [[T:_.+]] = copy _2 as
            // CHECK-NEXT: PtrMetadata(copy [[T]])
            let t3 = CastPtrToPtr::<_, *const (T, B)>(pa);
            let m3 = PtrMetadata(t3);

            // Both have tail `A`, yes optimize.
            // CHECK: [[T:_.+]] = copy _2 as
            // CHECK-NEXT: PtrMetadata(copy _2)
            let t4 = CastPtrToPtr::<_, *const (T, A)>(pa);
            let m4 = PtrMetadata(t4);

            // Tail `B` and tail `A`, do not optimize.
            // CHECK: [[T:_.+]] = copy _3 as
            // CHECK-NEXT: PtrMetadata(copy [[T]])
            let t5 = CastPtrToPtr::<_, *mut A>(pb);
            let m5 = PtrMetadata(t5);

            // Both have tail `B`, yes optimize.
            // CHECK: [[T:_.+]] = copy _3 as
            // CHECK-NEXT: PtrMetadata(copy _3)
            let t6 = CastPtrToPtr::<_, *mut B>(pb);
            let m6 = PtrMetadata(t6);

            Return()
        }
    }
}

fn cast_pointer_eq(p1: *mut u8, p2: *mut u32, p3: *mut u32, p4: *mut [u32]) {
    // CHECK-LABEL: fn cast_pointer_eq
    // CHECK: debug p1 => [[P1:_1]];
    // CHECK: debug p2 => [[P2:_2]];
    // CHECK: debug p3 => [[P3:_3]];
    // CHECK: debug p4 => [[P4:_4]];

    // CHECK: [[M1:_.+]] = copy [[P1]] as *const u32 (PtrToPtr);
    // CHECK: [[M2:_.+]] = copy [[P2]] as *const u32 (PtrToPtr);
    // CHECK: [[M3:_.+]] = copy [[P3]] as *const u32 (PtrToPtr);
    // CHECK: [[M4:_.+]] = copy [[P4]] as *const u32 (PtrToPtr);
    let m1 = p1 as *const u32;
    let m2 = p2 as *const u32;
    let m3 = p3 as *const u32;
    let m4 = p4 as *const u32;

    // CHECK-NOT: Eq
    // CHECK: Eq(copy [[M1]], copy [[M2]])
    // CHECK-NOT: Eq
    // CHECK: Eq(copy [[P2]], copy [[P3]])
    // CHECK-NOT: Eq
    // CHECK: Eq(copy [[M3]], copy [[M4]])
    // CHECK-NOT: Eq
    let eq_different_thing = m1 == m2;
    let eq_optimize = m2 == m3;
    let eq_thin_fat = m3 == m4;

    // CHECK: _0 = const ();
}

unsafe fn aggregate_struct_then_transmute(id: u16, thin: *const u8) {
    // CHECK: opaque::<u16>(copy _1)
    let a = MyId(id);
    opaque(std::intrinsics::transmute::<_, u16>(a));

    // CHECK: opaque::<u16>(copy _1)
    let b = TypedId::<String>(id, PhantomData);
    opaque(std::intrinsics::transmute::<_, u16>(b));

    // CHECK: opaque::<u16>(copy _1)
    let c = Err::<Never, u16>(id);
    opaque(std::intrinsics::transmute::<_, u16>(c));

    // CHECK: [[TEMP1:_[0-9]+]] = Option::<u16>::Some(copy _1);
    // CHECK: [[TEMP2:_[0-9]+]] = copy [[TEMP1]] as u32 (Transmute);
    // CHECK: opaque::<u32>(move [[TEMP2]])
    let d = Some(id);
    opaque(std::intrinsics::transmute::<_, u32>(d));

    // Still need the transmute, but the aggregate can be skipped
    // CHECK: [[TEMP:_[0-9]+]] = copy _1 as i16 (Transmute);
    // CHECK: opaque::<i16>(move [[TEMP]])
    let e = MyId(id);
    opaque(std::intrinsics::transmute::<_, i16>(e));

    // CHECK: [[PAIR:_[0-9]+]] = Pair(copy _1, copy _1);
    // CHECK: [[TEMP:_[0-9]+]] = copy [[PAIR]] as u32 (Transmute);
    // CHECK: opaque::<u32>(move [[TEMP]])
    struct Pair(u16, u16);
    let f = Pair(id, id);
    opaque(std::intrinsics::transmute::<_, u32>(f));

    // CHECK: [[TEMP:_[0-9]+]] = copy [[PAIR]] as u16 (Transmute);
    // CHECK: opaque::<u16>(move [[TEMP]])
    let g = Pair(id, id);
    opaque(std::intrinsics::transmute_unchecked::<_, u16>(g));

    // CHECK: opaque::<u16>(copy _1)
    let h = (id,);
    opaque(std::intrinsics::transmute::<_, u16>(h));

    // CHECK: opaque::<u16>(copy _1)
    let i = [id];
    opaque(std::intrinsics::transmute::<_, u16>(i));

    // CHECK: opaque::<*const u8>(copy _2)
    let j: *const i32 = std::intrinsics::aggregate_raw_ptr(thin, ());
    opaque(std::intrinsics::transmute::<_, *const u8>(j));
}

unsafe fn transmute_then_transmute_again(a: u32, c: char) {
    // CHECK: [[TEMP1:_[0-9]+]] = copy _1 as char (Transmute);
    // CHECK: [[TEMP2:_[0-9]+]] = copy [[TEMP1]] as i32 (Transmute);
    // CHECK: opaque::<i32>(move [[TEMP2]])
    let x = std::intrinsics::transmute::<u32, char>(a);
    opaque(std::intrinsics::transmute::<char, i32>(x));

    // CHECK: [[TEMP:_[0-9]+]] = copy _2 as i32 (Transmute);
    // CHECK: opaque::<i32>(move [[TEMP]])
    let x = std::intrinsics::transmute::<char, u32>(c);
    opaque(std::intrinsics::transmute::<u32, i32>(x));
}

// Transmuting can skip a pointer cast so long as it wasn't a fat-to-thin cast.
unsafe fn cast_pointer_then_transmute(thin: *mut u32, fat: *mut [u8]) {
    // CHECK-LABEL: fn cast_pointer_then_transmute

    // CHECK: [[UNUSED:_.+]] = copy _1 as *const () (PtrToPtr);
    // CHECK: = copy _1 as usize (Transmute);
    let thin_addr: usize = std::intrinsics::transmute(thin as *const ());

    // CHECK: [[TEMP2:_.+]] = copy _2 as *const () (PtrToPtr);
    // CHECK: = move [[TEMP2]] as usize (Transmute);
    let fat_addr: usize = std::intrinsics::transmute(fat as *const ());
}

unsafe fn transmute_then_cast_pointer(addr: usize, fat: *mut [u8]) {
    // CHECK-LABEL: fn transmute_then_cast_pointer

    // This is roughly what `NonNull::dangling` does
    // CHECK: [[CPTR:_.+]] = copy _1 as *const u8 (Transmute);
    // CHECK: takes_const_ptr::<u8>(move [[CPTR]])
    let p: *mut u8 = std::intrinsics::transmute(addr);
    takes_const_ptr(p);

    // This cast is fat-to-thin, so can't be merged with the transmute
    // CHECK: [[FAT:_.+]] = move {{.+}} as *const [i32] (Transmute);
    // CHECK: [[THIN:_.+]] = copy [[FAT]] as *const i32 (PtrToPtr);
    // CHECK: takes_const_ptr::<i32>(move [[THIN]])
    let q = std::intrinsics::transmute::<&mut [i32], *const [i32]>(&mut [1, 2, 3]);
    takes_const_ptr(q as *const i32);

    // CHECK: [[TPTR:_.+]] = copy _2 as *const u8 (PtrToPtr);
    // CHECK: takes_const_ptr::<u8>(move [[TPTR]])
    let w = std::intrinsics::transmute::<*mut [u8], *const [u8]>(fat);
    takes_const_ptr(w as *const u8);
}

#[custom_mir(dialect = "analysis")]
fn remove_casts_must_change_both_sides(mut_a: &*mut u8, mut_b: *mut u8) -> bool {
    // CHECK-LABEL: fn remove_casts_must_change_both_sides(
    mir! {
        // We'd like to remove these casts, but we can't change *both* of them
        // to be locals, so make sure we don't change one without the other, as
        // that would be a type error.
        {
            // CHECK: [[A:_.+]] = copy (*_1) as *const u8 (PtrToPtr);
            let a = *mut_a as *const u8;
            // CHECK: [[B:_.+]] = copy _2 as *const u8 (PtrToPtr);
            let b = mut_b as *const u8;
            // CHECK: _0 = Eq(copy [[A]], copy [[B]]);
            RET = a == b;
            Return()
        }
    }
}

fn main() {
    subexpression_elimination(2, 4, 5);
    wrap_unwrap(5);
    repeated_index::<u32, 7>(5, 3);
    unary(i64::MIN);
    arithmetic(5);
    comparison(5, 6);
    arithmetic_checked(5);
    arithmetic_float(5.);
    cast();
    multiple_branches(true, 5, 9);
    references(5);
    dereferences(&mut 5, &6, &S(7));
    slices();
    let (direct, indirect) = duplicate_slice();
    assert_eq!(direct, indirect);
    repeat();
    fn_pointers();
    indirect_static();
    constant_index_overflow(&[5, 3]);
    wide_ptr_provenance();
    wide_ptr_integer();
    borrowed(5);
    non_freeze(5);
    slice_const_length(&[1]);
    meta_of_ref_to_slice(&42);
    slice_from_raw_parts_as_ptr(&123, 456);
}

#[inline(never)]
fn opaque(_: impl Sized) {}

#[inline(never)]
fn identity<T>(x: T) -> T {
    x
}

#[inline(never)]
fn takes_const_ptr<T>(_: *const T) {}

#[repr(transparent)]
#[rustc_layout_scalar_valid_range_end(55555)]
struct MyId(u16);

#[repr(transparent)]
struct TypedId<T>(u16, PhantomData<T>);

enum Never {}

// EMIT_MIR gvn.subexpression_elimination.GVN.diff
// EMIT_MIR gvn.wrap_unwrap.GVN.diff
// EMIT_MIR gvn.repeated_index.GVN.diff
// EMIT_MIR gvn.unary.GVN.diff
// EMIT_MIR gvn.arithmetic.GVN.diff
// EMIT_MIR gvn.comparison.GVN.diff
// EMIT_MIR gvn.arithmetic_checked.GVN.diff
// EMIT_MIR gvn.arithmetic_float.GVN.diff
// EMIT_MIR gvn.cast.GVN.diff
// EMIT_MIR gvn.multiple_branches.GVN.diff
// EMIT_MIR gvn.references.GVN.diff
// EMIT_MIR gvn.dereferences.GVN.diff
// EMIT_MIR gvn.slices.GVN.diff
// EMIT_MIR gvn.duplicate_slice.GVN.diff
// EMIT_MIR gvn.repeat.GVN.diff
// EMIT_MIR gvn.fn_pointers.GVN.diff
// EMIT_MIR gvn.indirect_static.GVN.diff
// EMIT_MIR gvn.constant_index_overflow.GVN.diff
// EMIT_MIR gvn.wide_ptr_provenance.GVN.diff
// EMIT_MIR gvn.wide_ptr_same_provenance.GVN.diff
// EMIT_MIR gvn.wide_ptr_integer.GVN.diff
// EMIT_MIR gvn.borrowed.GVN.diff
// EMIT_MIR gvn.non_freeze.GVN.diff
// EMIT_MIR gvn.slice_const_length.GVN.diff
// EMIT_MIR gvn.meta_of_ref_to_slice.GVN.diff
// EMIT_MIR gvn.slice_from_raw_parts_as_ptr.GVN.diff
// EMIT_MIR gvn.casts_before_aggregate_raw_ptr.GVN.diff
// EMIT_MIR gvn.manual_slice_mut_len.GVN.diff
// EMIT_MIR gvn.array_len.GVN.diff
// EMIT_MIR gvn.dedup_multiple_bounds_checks_lengths.GVN.diff
// EMIT_MIR gvn.generic_cast_metadata.GVN.diff
// EMIT_MIR gvn.cast_pointer_eq.GVN.diff
// EMIT_MIR gvn.aggregate_struct_then_transmute.GVN.diff
// EMIT_MIR gvn.transmute_then_transmute_again.GVN.diff
// EMIT_MIR gvn.cast_pointer_then_transmute.GVN.diff
// EMIT_MIR gvn.transmute_then_cast_pointer.GVN.diff
// EMIT_MIR gvn.remove_casts_must_change_both_sides.GVN.diff
