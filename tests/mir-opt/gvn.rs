// skip-filecheck
// unit-test: GVN
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(raw_ref_op)]
#![feature(rustc_attrs)]
#![allow(unconditional_panic)]

struct S<T>(T);

fn subexpression_elimination(x: u64, y: u64, mut z: u64) {
    opaque(x + y);
    opaque(x * y);
    opaque(x - y);
    opaque(x / y);
    opaque(x % y);
    opaque(x & y);
    opaque(x | y);
    opaque(x ^ y);
    opaque(x << y);
    opaque(x >> y);
    opaque(x as u32);
    opaque(x as f32);
    opaque(S(x));
    opaque(S(x).0);

    // Those are duplicates to substitute somehow.
    opaque((x + y) + z);
    opaque((x * y) + z);
    opaque((x - y) + z);
    opaque((x / y) + z);
    opaque((x % y) + z);
    opaque((x & y) + z);
    opaque((x | y) + z);
    opaque((x ^ y) + z);
    opaque((x << y) + z);
    opaque((x >> y) + z);
    opaque(S(x));
    opaque(S(x).0);

    // We can substitute through an immutable reference too.
    let a = &z;
    opaque(*a + x);
    opaque(*a + x);

    // But not through a mutable reference or a pointer.
    let b = &mut z;
    opaque(*b + x);
    opaque(*b + x);
    unsafe {
        let c = &raw const z;
        opaque(*c + x);
        opaque(*c + x);
        let d = &raw mut z;
        opaque(*d + x);
        opaque(*d + x);
    }

    // We can substitute again, but not with the earlier computations.
    // Important: `e` is not `a`!
    let e = &z;
    opaque(*e + x);
    opaque(*e + x);

}

fn wrap_unwrap<T: Copy>(x: T) -> T {
    match Some(x) {
        Some(y) => y,
        None => panic!(),
    }
}

fn repeated_index<T: Copy, const N: usize>(x: T, idx: usize) {
    let a = [x; N];
    opaque(a[0]);
    opaque(a[idx]);
}

fn arithmetic(x: u64) {
    opaque(x + 0);
    opaque(x - 0);
    opaque(x * 0);
    opaque(x * 1);
    opaque(x / 0);
    opaque(x / 1);
    opaque(0 / x);
    opaque(1 / x);
    opaque(x % 0);
    opaque(x % 1);
    opaque(0 % x);
    opaque(1 % x);
    opaque(x & 0);
    opaque(x | 0);
    opaque(x ^ 0);
    opaque(x >> 0);
    opaque(x << 0);
}

#[rustc_inherit_overflow_checks]
fn arithmetic_checked(x: u64) {
    opaque(x + 0);
    opaque(x - 0);
    opaque(x * 0);
    opaque(x * 1);
    opaque(x / 0);
    opaque(x / 1);
    opaque(0 / x);
    opaque(1 / x);
    opaque(x % 0);
    opaque(x % 1);
    opaque(0 % x);
    opaque(1 % x);
    opaque(x & 0);
    opaque(x | 0);
    opaque(x ^ 0);
    opaque(x >> 0);
    opaque(x << 0);
}

fn arithmetic_float(x: f64) {
    opaque(x + 0.);
    opaque(x - 0.);
    opaque(x * 0.);
    opaque(x / 0.);
    opaque(0. / x);
    opaque(x % 0.);
    opaque(0. % x);
    // Those are not simplifiable to `true`/`false`, thanks to NaNs.
    opaque(x == x);
    opaque(x != x);
}

fn cast() {
    let i = 1_i64;
    let u = 1_u64;
    let f = 1_f64;
    opaque(i as u8);
    opaque(i as u16);
    opaque(i as u32);
    opaque(i as u64);
    opaque(i as i8);
    opaque(i as i16);
    opaque(i as i32);
    opaque(i as i64);
    opaque(i as f32);
    opaque(i as f64);
    opaque(u as u8);
    opaque(u as u16);
    opaque(u as u32);
    opaque(u as u64);
    opaque(u as i8);
    opaque(u as i16);
    opaque(u as i32);
    opaque(u as i64);
    opaque(u as f32);
    opaque(u as f64);
    opaque(f as u8);
    opaque(f as u16);
    opaque(f as u32);
    opaque(f as u64);
    opaque(f as i8);
    opaque(f as i16);
    opaque(f as i32);
    opaque(f as i64);
    opaque(f as f32);
    opaque(f as f64);
}

fn multiple_branches(t: bool, x: u8, y: u8) {
    if t {
        opaque(x + y); // a
        opaque(x + y); // should reuse a
    } else {
        opaque(x + y); // b
        opaque(x + y); // shoud reuse b
    }
    opaque(x + y); // c
    if t {
        opaque(x + y); // should reuse c
    } else {
        opaque(x + y); // should reuse c
    }
}

fn references(mut x: impl Sized) {
    opaque(&x);
    opaque(&x); // should not reuse a
    opaque(&mut x);
    opaque(&mut x); // should not reuse a
    opaque(&raw const x);
    opaque(&raw const x); // should not reuse a
    opaque(&raw mut x);
    opaque(&raw mut x); // should not reuse a
}

fn dereferences(t: &mut u32, u: &impl Copy, s: &S<u32>) {
    opaque(*t);
    opaque(*t); // this cannot reuse a, as x is &mut.
    let z = &raw const *t;
    unsafe { opaque(*z) };
    unsafe { opaque(*z) }; // this cannot reuse a, as x is *const.
    let z = &raw mut *t;
    unsafe { opaque(*z) };
    unsafe { opaque(*z) }; // this cannot reuse a, as x is *mut.
    let z = &*t;
    opaque(*z);
    opaque(*z); // this can reuse, as `z` is immutable ref, Freeze and Copy.
    opaque(&*z); // but not for a reborrow.
    opaque(*u);
    opaque(*u); // this cannot reuse, as `z` is not Freeze.
    opaque(s.0);
    opaque(s.0); // *s is not Copy, by (*s).0 is, so we can reuse.
}

fn slices() {
    let s = "my favourite slice"; // This is a `Const::Slice` in MIR.
    opaque(s);
    let t = s; // This should be the same pointer, so cannot be a `Const::Slice`.
    opaque(t);
    assert_eq!(s.as_ptr(), t.as_ptr());
    let u = unsafe { std::mem::transmute::<&str, &[u8]>(s) };
    opaque(u);
    assert_eq!(s.as_ptr(), u.as_ptr());
}

fn main() {
    subexpression_elimination(2, 4, 5);
    wrap_unwrap(5);
    repeated_index::<u32, 7>(5, 3);
    arithmetic(5);
    arithmetic_checked(5);
    arithmetic_float(5.);
    cast();
    multiple_branches(true, 5, 9);
    references(5);
    dereferences(&mut 5, &6, &S(7));
    slices();
}

#[inline(never)]
fn opaque(_: impl Sized) {}

// EMIT_MIR gvn.subexpression_elimination.GVN.diff
// EMIT_MIR gvn.wrap_unwrap.GVN.diff
// EMIT_MIR gvn.repeated_index.GVN.diff
// EMIT_MIR gvn.arithmetic.GVN.diff
// EMIT_MIR gvn.arithmetic_checked.GVN.diff
// EMIT_MIR gvn.arithmetic_float.GVN.diff
// EMIT_MIR gvn.cast.GVN.diff
// EMIT_MIR gvn.multiple_branches.GVN.diff
// EMIT_MIR gvn.references.GVN.diff
// EMIT_MIR gvn.dereferences.GVN.diff
// EMIT_MIR gvn.slices.GVN.diff
