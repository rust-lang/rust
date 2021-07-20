// compile-flags: -Zmir-opt-level=4 -Zinline-mir=yes

struct Tag(usize);

#[repr(C)]
struct S(Tag, Tag, Tag);

impl Drop for Tag {
    #[inline(never)]
    fn drop(&mut self) {}
}

// EMIT_MIR sroa.dropping.FlattenLocals.diff
pub fn dropping() {
    S(Tag(0), Tag(1), Tag(2)).1;
}

// EMIT_MIR sroa.enums.FlattenLocals.diff
pub fn enums(a: usize) -> usize {
    if let Some(a) = Some(a) { a } else { 0 }
}

// EMIT_MIR sroa.structs.FlattenLocals.diff
pub fn structs(a: f32) -> f32 {
    struct U {
        _foo: usize,
        a: f32,
    }

    U { _foo: 0, a }.a
}

// EMIT_MIR sroa.unions.FlattenLocals.diff
pub fn unions(a: f32) -> u32 {
    union Repr {
        f: f32,
        u: u32,
    }
    unsafe { Repr { f: a }.u }
}

fn main() {
    dropping();
    enums(5);
    structs(5.);
    unions(5.);
}
