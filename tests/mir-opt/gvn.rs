// unit-test: GVN
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// only-64bit

#![feature(raw_ref_op)]
#![feature(rustc_attrs)]
#![feature(custom_mir)]
#![feature(core_intrinsics)]
#![allow(unconditional_panic)]

use std::intrinsics::mir::*;
use std::mem::transmute;

struct S<T>(T);

fn subexpression_elimination(x: u64, y: u64, mut z: u64) {
    // CHECK-LABEL: fn subexpression_elimination(

    // CHECK: [[add:_.*]] = Add(_1, _2);
    // CHECK: opaque::<u64>([[add]])
    opaque(x + y);
    // CHECK: [[mul:_.*]] = Mul(_1, _2);
    // CHECK: opaque::<u64>([[mul]])
    opaque(x * y);
    // CHECK: [[sub:_.*]] = Sub(_1, _2);
    // CHECK: opaque::<u64>([[sub]])
    opaque(x - y);
    // CHECK: [[div:_.*]] = Div(_1, _2);
    // CHECK: opaque::<u64>([[div]])
    opaque(x / y);
    // CHECK: [[rem:_.*]] = Rem(_1, _2);
    // CHECK: opaque::<u64>([[rem]])
    opaque(x % y);
    // CHECK: [[and:_.*]] = BitAnd(_1, _2);
    // CHECK: opaque::<u64>([[and]])
    opaque(x & y);
    // CHECK: [[or:_.*]] = BitOr(_1, _2);
    // CHECK: opaque::<u64>([[or]])
    opaque(x | y);
    // CHECK: [[xor:_.*]] = BitXor(_1, _2);
    // CHECK: opaque::<u64>([[xor]])
    opaque(x ^ y);
    // CHECK: [[shl:_.*]] = Shl(_1, _2);
    // CHECK: opaque::<u64>([[shl]])
    opaque(x << y);
    // CHECK: [[shr:_.*]] = Shr(_1, _2);
    // CHECK: opaque::<u64>([[shr]])
    opaque(x >> y);
    // CHECK: [[int:_.*]] = _1 as u32 (IntToInt);
    // CHECK: opaque::<u32>([[int]])
    opaque(x as u32);
    // CHECK: [[float:_.*]] = _1 as f32 (IntToFloat);
    // CHECK: opaque::<f32>([[float]])
    opaque(x as f32);
    // CHECK: [[wrap:_.*]] = S::<u64>(_1);
    // CHECK: opaque::<S<u64>>([[wrap]])
    opaque(S(x));
    // CHECK: opaque::<u64>(_1)
    opaque(S(x).0);

    // Those are duplicates to substitute somehow.
    // CHECK: opaque::<u64>([[add]])
    opaque(x + y);
    // CHECK: opaque::<u64>([[mul]])
    opaque(x * y);
    // CHECK: opaque::<u64>([[sub]])
    opaque(x - y);
    // CHECK: opaque::<u64>([[div]])
    opaque(x / y);
    // CHECK: opaque::<u64>([[rem]])
    opaque(x % y);
    // CHECK: opaque::<u64>([[and]])
    opaque(x & y);
    // CHECK: opaque::<u64>([[or]])
    opaque(x | y);
    // CHECK: opaque::<u64>([[xor]])
    opaque(x ^ y);
    // CHECK: opaque::<u64>([[shl]])
    opaque(x << y);
    // CHECK: opaque::<u64>([[shr]])
    opaque(x >> y);
    // CHECK: opaque::<u32>([[int]])
    opaque(x as u32);
    // CHECK: opaque::<f32>([[float]])
    opaque(x as f32);
    // CHECK: opaque::<S<u64>>([[wrap]])
    opaque(S(x));
    // CHECK: opaque::<u64>(_1)
    opaque(S(x).0);

    // We can substitute through a complex expression.
    // CHECK: [[compound:_.*]] = Sub([[mul]], _2);
    // CHECK: opaque::<u64>([[compound]])
    // CHECK: opaque::<u64>([[compound]])
    opaque((x * y) - y);
    opaque((x * y) - y);

    // We can substitute through an immutable reference too.
    // CHECK: [[ref:_.*]] = &_3;
    // CHECK: [[deref:_.*]] = (*[[ref]]);
    // CHECK: [[addref:_.*]] = Add([[deref]], _1);
    // CHECK: opaque::<u64>([[addref]])
    // CHECK: opaque::<u64>([[addref]])
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

    // We can substitute again, but not with the earlier computations.
    // Important: `e` is not `a`!
    // CHECK: [[ref2:_.*]] = &_3;
    // CHECK: [[deref2:_.*]] = (*[[ref2]]);
    // CHECK: [[addref2:_.*]] = Add([[deref2]], _1);
    // CHECK: opaque::<u64>([[addref2]])
    // CHECK: opaque::<u64>([[addref2]])
    let e = &z;
    opaque(*e + x);
    opaque(*e + x);
}

fn wrap_unwrap<T: Copy>(x: T) -> T {
    // CHECK-LABEL: fn wrap_unwrap(
    // CHECK: [[some:_.*]] = Option::<T>::Some(_1);
    // CHECK: switchInt(const 1_isize)
    // CHECK: _0 = _1;
    match Some(x) {
        Some(y) => y,
        None => panic!(),
    }
}

fn repeated_index<T: Copy, const N: usize>(x: T, idx: usize) {
    // CHECK-LABEL: fn repeated_index(
    // CHECK: [[a:_.*]] = [_1; N];
    let a = [x; N];
    // CHECK: opaque::<T>(_1)
    opaque(a[0]);
    // CHECK: opaque::<T>(_1)
    opaque(a[idx]);
}

fn unary(x: i64) {
    // CHECK-LABEL: fn unary(
    // CHECK: opaque::<i64>(_1)
    opaque(--x); // This is `x`.

    // CHECK: [[b:_.*]] = Lt(_1, const 13_i64);
    // CHECK: opaque::<bool>([[b]])
    let b = x < 13;
    opaque(!!b); // This is `b`.

    // Both lines should test the same thing.
    // CHECK: [[c:_.*]] = Ne(_1, const 15_i64);
    // CHECK: opaque::<bool>([[c]])
    // CHECK: opaque::<bool>([[c]])
    opaque(x != 15);
    opaque(!(x == 15));

    // Both lines should test the same thing.
    // CHECK: [[d:_.*]] = Eq(_1, const 35_i64);
    // CHECK: opaque::<bool>([[d]])
    // CHECK: opaque::<bool>([[d]])
    opaque(x == 35);
    opaque(!(x != 35));
}

/// Verify symbolic integer arithmetic simplifications.
fn arithmetic(x: u64) {
    // CHECK-LABEL: fn arithmetic(
    // CHECK: opaque::<u64>(_1)
    opaque(x + 0);
    // CHECK: opaque::<u64>(_1)
    opaque(x - 0);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x - x);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x * 0);
    // CHECK: opaque::<u64>(_1)
    opaque(x * 1);
    // CHECK: assert(!const true, "attempt to divide `{}` by zero",
    // CHECK: [[div0:_.*]] = Div(_1, const 0_u64);
    // CHECK: opaque::<u64>(move [[div0]])
    opaque(x / 0);
    // CHECK: opaque::<u64>(_1)
    opaque(x / 1);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(0 / x);
    // CHECK: [[odiv:_.*]] = Div(const 1_u64, _1);
    // CHECK: opaque::<u64>(move [[odiv]])
    opaque(1 / x);
    // CHECK: assert(!const true, "attempt to calculate the remainder of `{}` with a divisor of zero"
    // CHECK: [[rem0:_.*]] = Rem(_1, const 0_u64);
    // CHECK: opaque::<u64>(move [[rem0]])
    opaque(x % 0);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x % 1);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(0 % x);
    // CHECK: [[orem:_.*]] = Rem(const 1_u64, _1);
    // CHECK: opaque::<u64>(move [[orem]])
    opaque(1 % x);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x & 0);
    // CHECK: opaque::<u64>(_1)
    opaque(x & u64::MAX);
    // CHECK: opaque::<u64>(_1)
    opaque(x | 0);
    // CHECK: opaque::<u64>(const u64::MAX)
    opaque(x | u64::MAX);
    // CHECK: opaque::<u64>(_1)
    opaque(x ^ 0);
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x ^ x);
    // CHECK: opaque::<u64>(_1)
    opaque(x >> 0);
    // CHECK: opaque::<u64>(_1)
    opaque(x << 0);
}

fn comparison(x: u64, y: u64) {
    // CHECK-LABEL: fn comparison(
    // CHECK: opaque::<bool>(const true)
    opaque(x == x);
    // CHECK: opaque::<bool>(const false)
    opaque(x != x);
    // CHECK: [[eqxy:_.*]] = Eq(_1, _2);
    // CHECK: opaque::<bool>(move [[eqxy]])
    opaque(x == y);
    // CHECK: [[nexy:_.*]] = Ne(_1, _2);
    // CHECK: opaque::<bool>(move [[nexy]])
    opaque(x != y);
}

/// Verify symbolic integer arithmetic simplifications on checked ops.
#[rustc_inherit_overflow_checks]
fn arithmetic_checked(x: u64) {
    // CHECK-LABEL: fn arithmetic_checked(
    // CHECK: assert(!const false,
    // CHECK: opaque::<u64>(_1)
    opaque(x + 0);
    // CHECK: assert(!const false,
    // CHECK: opaque::<u64>(_1)
    opaque(x - 0);
    // CHECK: assert(!const false,
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x - x);
    // CHECK: assert(!const false,
    // CHECK: opaque::<u64>(const 0_u64)
    opaque(x * 0);
    // CHECK: assert(!const false,
    // CHECK: opaque::<u64>(_1)
    opaque(x * 1);
}

/// Verify that we do not apply arithmetic simplifications on floats.
fn arithmetic_float(x: f64) {
    // CHECK-LABEL: fn arithmetic_float(
    // CHECK: [[add:_.*]] = Add(_1, const 0f64);
    // CHECK: opaque::<f64>(move [[add]])
    opaque(x + 0.);
    // CHECK: [[sub:_.*]] = Sub(_1, const 0f64);
    // CHECK: opaque::<f64>(move [[sub]])
    opaque(x - 0.);
    // CHECK: [[mul:_.*]] = Mul(_1, const 0f64);
    // CHECK: opaque::<f64>(move [[mul]])
    opaque(x * 0.);
    // CHECK: [[div0:_.*]] = Div(_1, const 0f64);
    // CHECK: opaque::<f64>(move [[div0]])
    opaque(x / 0.);
    // CHECK: [[zdiv:_.*]] = Div(const 0f64, _1);
    // CHECK: opaque::<f64>(move [[zdiv]])
    opaque(0. / x);
    // CHECK: [[rem0:_.*]] = Rem(_1, const 0f64);
    // CHECK: opaque::<f64>(move [[rem0]])
    opaque(x % 0.);
    // CHECK: [[zrem:_.*]] = Rem(const 0f64, _1);
    // CHECK: opaque::<f64>(move [[zrem]])
    opaque(0. % x);
    // Those are not simplifiable to `true`/`false`, thanks to NaNs.
    // CHECK: [[eq:_.*]] = Eq(_1, _1);
    // CHECK: opaque::<bool>(move [[eq]])
    opaque(x == x);
    // CHECK: [[ne:_.*]] = Ne(_1, _1);
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
    // CHECK: switchInt(_1) -> [0: [[bbf:bb.*]], otherwise: [[bbt:bb.*]]];
    if t {
        // CHECK: [[bbt]]: {
        // CHECK: [[a:_.*]] = Add(_2, _3);
        // CHECK: opaque::<u8>([[a]])
        // CHECK: opaque::<u8>([[a]])
        // CHECK: goto -> [[bbc:bb.*]];
        opaque(x + y);
        opaque(x + y);
    } else {
        // CHECK: [[bbf]]: {
        // CHECK: [[b:_.*]] = Add(_2, _3);
        // CHECK: opaque::<u8>([[b]])
        // CHECK: opaque::<u8>([[b]])
        // CHECK: goto -> [[bbc:bb.*]];
        opaque(x + y);
        opaque(x + y);
    }
    // Neither `a` nor `b` dominate `c`, so we cannot reuse any of them.
    // CHECK: [[bbc]]: {
    // CHECK: [[c:_.*]] = Add(_2, _3);
    // CHECK: opaque::<u8>([[c]])
    opaque(x + y);

    // `c` dominates both calls, so we can reuse it.
    if t {
        // CHECK: opaque::<u8>([[c]])
        opaque(x + y);
    } else {
        // CHECK: opaque::<u8>([[c]])
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
    // CHECK: [[st1:_.*]] = (*_1);
    // CHECK: opaque::<u32>(move [[st1]])
    // CHECK: [[st2:_.*]] = (*_1);
    // CHECK: opaque::<u32>(move [[st2]])
    opaque(*t);
    opaque(*t);

    // Do not reuse dereferences of `*const`.
    // CHECK: [[raw:_.*]] = &raw const (*_1);
    // CHECK: [[st3:_.*]] = (*[[raw]]);
    // CHECK: opaque::<u32>(move [[st3]])
    // CHECK: [[st4:_.*]] = (*[[raw]]);
    // CHECK: opaque::<u32>(move [[st4]])
    let z = &raw const *t;
    unsafe { opaque(*z) };
    unsafe { opaque(*z) };

    // Do not reuse dereferences of `*mut`.
    // CHECK: [[ptr:_.*]] = &raw mut (*_1);
    // CHECK: [[st5:_.*]] = (*[[ptr]]);
    // CHECK: opaque::<u32>(move [[st5]])
    // CHECK: [[st6:_.*]] = (*[[ptr]]);
    // CHECK: opaque::<u32>(move [[st6]])
    let z = &raw mut *t;
    unsafe { opaque(*z) };
    unsafe { opaque(*z) };

    // We can reuse dereferences of `&Freeze`.
    // CHECK: [[ref:_.*]] = &(*_1);
    // CHECK: [[st7:_.*]] = (*[[ref]]);
    // CHECK: opaque::<u32>([[st7]])
    // CHECK: opaque::<u32>([[st7]])
    let z = &*t;
    opaque(*z);
    opaque(*z);
    // But not in reborrows.
    // CHECK: [[reborrow:_.*]] = &(*[[ref]]);
    // CHECK: opaque::<&u32>(move [[reborrow]])
    opaque(&*z);

    // `*u` is not Freeze, so we cannot reuse.
    // CHECK: [[st8:_.*]] = (*_2);
    // CHECK: opaque::<impl Copy>(move [[st8]])
    // CHECK: [[st9:_.*]] = (*_2);
    // CHECK: opaque::<impl Copy>(move [[st9]])
    opaque(*u);
    opaque(*u);

    // `*s` is not Copy, by `(*s).0` is, so we can reuse.
    // CHECK: [[st10:_.*]] = ((*_3).0: u32);
    // CHECK: opaque::<u32>([[st10]])
    // CHECK: opaque::<u32>([[st10]])
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
    mir!(
        let au: u128;
        let bu: u128;
        let cu: u128;
        let du: u128;
        let c: &str;
        let d: &str;
        {
            // CHECK: [[a:_.*]] = (const "a",);
            // CHECK: [[au:_.*]] = ([[a]].0: &str) as u128 (Transmute);
            let a = ("a",);
            Call(au = transmute::<_, u128>(a.0), ReturnTo(bb1), UnwindContinue())
        }
        bb1 = {
            // CHECK: [[c:_.*]] = identity::<&str>(([[a]].0: &str))
            Call(c = identity(a.0), ReturnTo(bb2), UnwindContinue())
        }
        bb2 = {
            // CHECK: [[cu:_.*]] = [[c]] as u128 (Transmute);
            Call(cu = transmute::<_, u128>(c), ReturnTo(bb3), UnwindContinue())
        }
        bb3 = {
            // This slice is different from `a.0`. Hence `bu` is not `au`.
            // CHECK: [[b:_.*]] = const "a";
            // CHECK: [[bu:_.*]] = [[b]] as u128 (Transmute);
            let b = "a";
            Call(bu = transmute::<_, u128>(b), ReturnTo(bb4), UnwindContinue())
        }
        bb4 = {
            // This returns a copy of `b`, which is not `a`.
            // CHECK: [[d:_.*]] = identity::<&str>([[b]])
            Call(d = identity(b), ReturnTo(bb5), UnwindContinue())
        }
        bb5 = {
            // CHECK: [[du:_.*]] = [[d]] as u128 (Transmute);
            Call(du = transmute::<_, u128>(d), ReturnTo(bb6), UnwindContinue())
        }
        bb6 = {
            // `direct` must not fold to `true`, as `indirect` will not.
            // CHECK: = Eq([[au]], [[bu]]);
            // CHECK: = Eq([[cu]], [[du]]);
            let direct = au == bu;
            let indirect = cu == du;
            RET = (direct, indirect);
            Return()
        }
    )
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
    // CHECK: opaque::<fn(u8) -> u8>([[f]])
    let f = identity as fn(u8) -> u8;
    opaque(f);
    // CHECK: [[g:_.*]] = identity::<u8> as fn(u8) -> u8 (PointerCoercion(ReifyFnPointer
    // CHECK: opaque::<fn(u8) -> u8>([[g]])
    let g = identity as fn(u8) -> u8;
    opaque(g);

    // CHECK: [[cf:_.*]] = const {{.*}} as fn() (PointerCoercion(ClosureFnPointer
    // CHECK: opaque::<fn()>([[cf]])
    let closure = || {};
    let cf = closure as fn();
    opaque(cf);
    // CHECK: [[cg:_.*]] = const {{.*}} as fn() (PointerCoercion(ClosureFnPointer
    // CHECK: opaque::<fn()>([[cg]])
    let cg = closure as fn();
    opaque(cg);
}

/// Verify that we do not create a `ConstValue::Indirect` backed by a static's AllocId.
#[custom_mir(dialect = "analysis")]
fn indirect_static() {
    static A: Option<u8> = None;

    mir!({
        let ptr = Static(A);
        let out = Field::<u8>(Variant(*ptr, 1), 0);
        Return()
    })
}

/// Verify that having constant index `u64::MAX` does not yield to an overflow in rustc.
fn constant_index_overflow<T: Copy>(x: &[T]) {
    // CHECK-LABEL: fn constant_index_overflow(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];
    // CHECK: [[a]] = const usize::MAX;
    // CHECK-NOT: = (*_1)[{{.*}} of 0];
    // CHECK: [[b]] = (*_1)[[[a]]];
    // CHECK-NOT: = (*_1)[{{.*}} of 0];
    // CHECK: [[b]] = (*_1)[0 of 1];
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

    // CHECK: [[eqp:_.*]] = Eq([[a:_.*]], [[b:_.*]]);
    // CHECK: opaque::<bool>(move [[eqp]])
    opaque(a == b);
    // CHECK: [[nep:_.*]] = Ne([[a]], [[b]]);
    // CHECK: opaque::<bool>(move [[nep]])
    opaque(a != b);
    // CHECK: [[ltp:_.*]] = Lt([[a]], [[b]]);
    // CHECK: opaque::<bool>(move [[ltp]])
    opaque(a < b);
    // CHECK: [[lep:_.*]] = Le([[a]], [[b]]);
    // CHECK: opaque::<bool>(move [[lep]])
    opaque(a <= b);
    // CHECK: [[gtp:_.*]] = Gt([[a]], [[b]]);
    // CHECK: opaque::<bool>(move [[gtp]])
    opaque(a > b);
    // CHECK: [[gep:_.*]] = Ge([[a]], [[b]]);
    // CHECK: opaque::<bool>(move [[gep]])
    opaque(a >= b);
}

/// Both pointers come form the same allocation, so we could probably fold the comparisons.
fn wide_ptr_same_provenance() {
    // CHECK-LABEL: fn wide_ptr_same_provenance(
    let slice = &[1, 2];
    let a: *const dyn Send = &slice[0] as &dyn Send;
    let b: *const dyn Send = &slice[1] as &dyn Send;

    // CHECK: [[eqp:_.*]] = Eq([[a:_.*]], [[b:_.*]]);
    // CHECK: opaque::<bool>(move [[eqp]])
    opaque(a == b);
    // CHECK: [[nep:_.*]] = Ne([[a]], [[b]]);
    // CHECK: opaque::<bool>(move [[nep]])
    opaque(a != b);
    // CHECK: [[ltp:_.*]] = Lt([[a]], [[b]]);
    // CHECK: opaque::<bool>(move [[ltp]])
    opaque(a < b);
    // CHECK: [[lep:_.*]] = Le([[a]], [[b]]);
    // CHECK: opaque::<bool>(move [[lep]])
    opaque(a <= b);
    // CHECK: [[gtp:_.*]] = Gt([[a]], [[b]]);
    // CHECK: opaque::<bool>(move [[gtp]])
    opaque(a > b);
    // CHECK: [[gep:_.*]] = Ge([[a]], [[b]]);
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
}

#[inline(never)]
fn opaque(_: impl Sized) {}

#[inline(never)]
fn identity<T>(x: T) -> T {
    x
}

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
