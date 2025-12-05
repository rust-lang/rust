//@ test-mir-pass: ScalarReplacementOfAggregates
//@ compile-flags: -Cpanic=abort
//@ no-prefer-dynamic

struct Tag(usize);

#[repr(C)]
struct S(Tag, Tag, Tag);

impl Drop for Tag {
    #[inline(never)]
    fn drop(&mut self) {}
}

/// Check that SROA excludes structs with a `Drop` implementation.
pub fn dropping() {
    // CHECK-LABEL: fn dropping(

    // CHECK: [[aggregate:_[0-9]+]]: S;

    // CHECK: bb0: {
    // CHECK: [[aggregate]] = S
    S(Tag(0), Tag(1), Tag(2)).1;
}

/// Check that SROA excludes enums.
pub fn enums(a: usize) -> usize {
    // CHECK-LABEL: fn enums(

    // CHECK: [[enum:_[0-9]+]]: std::option::Option<usize>;

    // CHECK: bb0: {
    // CHECK: [[enum]] = Option::<usize>::Some
    // CHECK: _5 = copy (([[enum]] as Some).0: usize)
    // CHECK: _0 = copy _5
    if let Some(a) = Some(a) { a } else { 0 }
}

/// Check that SROA destructures `U`.
pub fn structs(a: f32) -> f32 {
    // CHECK-LABEL: fn structs(
    struct U {
        _foo: usize,
        a: f32,
    }
    // CHECK: [[ret:_0]]: f32;
    // CHECK: [[struct:_[0-9]+]]: structs::U;
    // CHECK: [[a_tmp:_[0-9]+]]: f32;
    // CHECK: [[foo:_[0-9]+]]: usize;
    // CHECK: [[a_ret:_[0-9]+]]: f32;

    // CHECK: bb0: {
    // CHECK-NOT: [[struct]]
    // CHECK: [[a_tmp]] = copy _1;
    // CHECK-NOT: [[struct]]
    // CHECK: [[foo]] = const 0_usize;
    // CHECK-NOT: [[struct]]
    // CHECK: [[a_ret]] = move [[a_tmp]];
    // CHECK-NOT: [[struct]]
    // CHECK: _0 = copy [[a_ret]];
    // CHECK-NOT: [[struct]]
    U { _foo: 0, a }.a
}

/// Check that SROA excludes unions.
pub fn unions(a: f32) -> u32 {
    // CHECK-LABEL: fn unions(
    union Repr {
        f: f32,
        u: u32,
    }
    // CHECK: [[union:_[0-9]+]]: unions::Repr;

    // CHECK: bb0: {
    // CHECK: [[union]] = Repr {
    // CHECK: _0 = copy ([[union]].1: u32)
    unsafe { Repr { f: a }.u }
}

#[derive(Copy, Clone)]
struct Foo {
    a: u8,
    b: (),
    c: &'static str,
    d: Option<isize>,
}

/// Check that non-escaping uses of a struct are destructured.
pub fn flat() {
    // CHECK-LABEL: fn flat(

    // CHECK: [[struct:_[0-9]+]]: Foo;

    // CHECK: bb0: {
    // CHECK: [[init_unit:_[0-9]+]] = ();
    // CHECK: [[init_opt_isize:_[0-9]+]] = Option::<isize>::Some

    // CHECK: [[destr_five:_[0-9]+]] = const 5_u8;
    // CHECK: [[destr_unit:_[0-9]+]] = move [[init_unit]];
    // CHECK: [[destr_a:_[0-9]+]] = const "a";
    // CHECK: [[destr_opt_isize:_[0-9]+]] = move [[init_opt_isize]];

    let Foo { a, b, c, d } = Foo { a: 5, b: (), c: "a", d: Some(-4) };
    let _ = a;
    let _ = b;
    let _ = c;
    let _ = d;
}

#[repr(C)]
struct Escaping {
    a: u32,
    b: u32,
    c: u32,
}

fn g() -> u32 {
    3
}

fn f(a: *const u32) {
    println!("{}", unsafe { *a.add(2) });
}

// `f` uses the `&e.a` to access `e.c`. This is UB according to Miri today; however,
// T-opsem has not finalized that decision and as such rustc should not rely on
// it. If SROA were to rely on it, it would be (almost) correct to turn `e` into
// three distinct locals - one for each field - and pass a reference to only one
// of them to `f`. However, this would lead to a miscompilation because `b` and `c`
// might no longer appear right after `a` in memory.
pub fn escaping() {
    // CHECK-LABEL: fn escaping(

    // CHECK: [[ptr:_[0-9]+]]: *const u32;
    // CHECK: [[ref:_[0-9]+]]: &u32;
    // CHECK: [[struct:_[0-9]+]]: Escaping;
    // CHECK: [[a:_[0-9]+]]: u32;

    // CHECK: bb0: {
    // CHECK: [[struct]] = Escaping {
    // CHECK: [[ref]] = &([[struct]].0
    // CHECK: [[ptr]] = &raw const (*[[ref]]);
    f(&Escaping { a: 1, b: 2, c: g() }.a);
}

/// Check that copies from an internal struct are destructured and reassigned to
/// the original struct.
fn copies(x: Foo) {
    // CHECK-LABEL: fn copies(

    // CHECK: [[external:_[0-9]+]]: Foo) ->
    // CHECK: [[internal:_[0-9]+]]: Foo;
    // CHECK: [[byte:_[0-9]+]]: u8;
    // CHECK: [[unit:_[0-9]+]]: ();
    // CHECK: [[str:_[0-9]+]]: &str;
    // CHECK: [[opt_isize:_[0-9]+]]: std::option::Option<isize>;

    // CHECK: bb0: {
    // CHECK: [[byte]] = copy ([[external]].0
    // CHECK: [[unit]] = copy ([[external]].1
    // CHECK: [[str]] = copy ([[external]].2
    // CHECK: [[opt_isize]] = copy ([[external]].3

    let y = x;
    let t = y.a;
    let u = y.c;
    let z = y;
    let a = z.b;
}

/// Check that copies from an internal struct are destructured and reassigned to
/// the original struct.
fn ref_copies(x: &Foo) {
    // CHECK-LABEL: fn ref_copies(

    // CHECK: [[external:_[0-9]+]]: &Foo) ->
    // CHECK: [[internal:_[0-9]+]]: Foo;
    // CHECK: [[byte:_[0-9]+]]: u8;
    // CHECK: [[unit:_[0-9]+]]: ();
    // CHECK: [[str:_[0-9]+]]: &str;
    // CHECK: [[opt_isize:_[0-9]+]]: std::option::Option<isize>;

    // CHECK: bb0: {
    // CHECK: [[byte]] = copy ((*[[external]]).0
    // CHECK: [[unit]] = copy ((*[[external]]).1
    // CHECK: [[str]] = copy ((*[[external]]).2
    // CHECK: [[opt_isize]] = copy ((*[[external]]).3

    let y = *x;
    let t = y.a;
    let u = y.c;
}

/// Check that deaggregated assignments from constants are placed after the constant's
/// assignment. Also check that copying field accesses from the copy of the constant are
/// reassigned to copy from the constant.
fn constant() {
    // CHECK-LABEL: constant(

    // CHECK: [[constant:_[0-9]+]]: (usize, u8);
    // CHECK: [[t:_[0-9]+]]: usize;
    // CHECK: [[u:_[0-9]+]]: u8;

    // CHECK: bb0: {
    // CHECK-NOT: [[constant]]
    // CHECK: [[constant]] = const
    // CHECK: [[t]] = move ([[constant]].0: usize)
    // CHECK: [[u]] = move ([[constant]].1: u8)
    const U: (usize, u8) = (5, 9);
    let y = U;
    let t = y.0;
    let u = y.1;
}

fn main() {
    // CHECK-LABEL: fn main(
    dropping();
    enums(5);
    structs(5.);
    unions(5.);
    flat();
    escaping();
    copies(Foo { a: 5, b: (), c: "a", d: Some(-4) });
    ref_copies(&Foo { a: 5, b: (), c: "a", d: Some(-4) });
    constant();
}

// EMIT_MIR structs.dropping.ScalarReplacementOfAggregates.diff
// EMIT_MIR structs.enums.ScalarReplacementOfAggregates.diff
// EMIT_MIR structs.structs.ScalarReplacementOfAggregates.diff
// EMIT_MIR structs.unions.ScalarReplacementOfAggregates.diff
// EMIT_MIR structs.flat.ScalarReplacementOfAggregates.diff
// EMIT_MIR structs.escaping.ScalarReplacementOfAggregates.diff
// EMIT_MIR structs.copies.ScalarReplacementOfAggregates.diff
// EMIT_MIR structs.ref_copies.ScalarReplacementOfAggregates.diff
// EMIT_MIR structs.constant.ScalarReplacementOfAggregates.diff
