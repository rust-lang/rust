// unit-test: ScalarReplacementOfAggregates
// compile-flags: -Cpanic=abort
// no-prefer-dynamic

struct Tag(usize);

#[repr(C)]
struct S(Tag, Tag, Tag);

impl Drop for Tag {
    #[inline(never)]
    fn drop(&mut self) {}
}

pub fn dropping() {
    S(Tag(0), Tag(1), Tag(2)).1;
}

pub fn enums(a: usize) -> usize {
    if let Some(a) = Some(a) { a } else { 0 }
}

pub fn structs(a: f32) -> f32 {
    struct U {
        _foo: usize,
        a: f32,
    }

    U { _foo: 0, a }.a
}

pub fn unions(a: f32) -> u32 {
    union Repr {
        f: f32,
        u: u32,
    }
    unsafe { Repr { f: a }.u }
}

#[derive(Copy, Clone)]
struct Foo {
    a: u8,
    b: (),
    c: &'static str,
    d: Option<isize>,
}

fn g() -> u32 {
    3
}

pub fn flat() {
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
    f(&Escaping { a: 1, b: 2, c: g() }.a);
}

fn copies(x: Foo) {
    let y = x;
    let t = y.a;
    let u = y.c;
    let z = y;
    let a = z.b;
}

fn ref_copies(x: &Foo) {
    let y = *x;
    let t = y.a;
    let u = y.c;
}

fn constant() {
    const U: (usize, u8) = (5, 9);
    let y = U;
    let t = y.0;
    let u = y.1;
}

fn main() {
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

// EMIT_MIR sroa.dropping.ScalarReplacementOfAggregates.diff
// EMIT_MIR sroa.enums.ScalarReplacementOfAggregates.diff
// EMIT_MIR sroa.structs.ScalarReplacementOfAggregates.diff
// EMIT_MIR sroa.unions.ScalarReplacementOfAggregates.diff
// EMIT_MIR sroa.flat.ScalarReplacementOfAggregates.diff
// EMIT_MIR sroa.escaping.ScalarReplacementOfAggregates.diff
// EMIT_MIR sroa.copies.ScalarReplacementOfAggregates.diff
// EMIT_MIR sroa.ref_copies.ScalarReplacementOfAggregates.diff
// EMIT_MIR sroa.constant.ScalarReplacementOfAggregates.diff
