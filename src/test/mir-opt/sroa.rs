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

// EMIT_MIR sroa.dropping.ScalarReplacementOfAggregates.diff
pub fn dropping() {
    S(Tag(0), Tag(1), Tag(2)).1;
}

// EMIT_MIR sroa.enums.ScalarReplacementOfAggregates.diff
pub fn enums(a: usize) -> usize {
    if let Some(a) = Some(a) { a } else { 0 }
}

// EMIT_MIR sroa.structs.ScalarReplacementOfAggregates.diff
pub fn structs(a: f32) -> f32 {
    struct U {
        _foo: usize,
        a: f32,
    }

    U { _foo: 0, a }.a
}

// EMIT_MIR sroa.unions.ScalarReplacementOfAggregates.diff
pub fn unions(a: f32) -> u32 {
    union Repr {
        f: f32,
        u: u32,
    }
    unsafe { Repr { f: a }.u }
}

struct Foo {
    a: u8,
    b: (),
    c: &'static str,
    d: Option<isize>,
}

fn g() -> u32 {
    3
}

// EMIT_MIR sroa.flat.ScalarReplacementOfAggregates.diff
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

// EMIT_MIR sroa.escaping.ScalarReplacementOfAggregates.diff
pub fn escaping() {
    // Verify this struct is not flattened.
    f(&Escaping { a: 1, b: 2, c: g() }.a);
}

fn main() {
    dropping();
    enums(5);
    structs(5.);
    unions(5.);
    flat();
    escaping();
}
