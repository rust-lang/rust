//@ compile-flags: -Zmir-opt-level=0
//@ edition: 2024
// skip-filecheck

// This test demonstrates how many user type annotations are recorded in MIR
// for various binding constructs. In particular, this makes it possible to see
// the number of duplicate user-type-annotation entries, and whether that
// number has changed.
//
// Duplicates are mostly harmless, other than being inefficient.
// "Unused" entries that are _not_ duplicates may nevertheless be necessary so
// that they are seen by MIR lifetime checks.

// EMIT_MIR user_type_annotations.let_uninit.built.after.mir
fn let_uninit() {
    let (x, y, z): (u32, u64, &'static char);
}

// EMIT_MIR user_type_annotations.let_uninit_bindless.built.after.mir
fn let_uninit_bindless() {
    let (_, _, _): (u32, u64, &'static char);
}

// EMIT_MIR user_type_annotations.let_init.built.after.mir
fn let_init() {
    let (x, y, z): (u32, u64, &'static char) = (7, 12, &'u');
}

// EMIT_MIR user_type_annotations.let_init_bindless.built.after.mir
fn let_init_bindless() {
    let (_, _, _): (u32, u64, &'static char) = (7, 12, &'u');
}

// EMIT_MIR user_type_annotations.let_else.built.after.mir
fn let_else() {
    let (x, y, z): (u32, u64, &'static char) = (7, 12, &'u') else { unreachable!() };
}

// EMIT_MIR user_type_annotations.let_else_bindless.built.after.mir
fn let_else_bindless() {
    let (_, _, _): (u32, u64, &'static char) = (7, 12, &'u') else { unreachable!() };
}

trait MyTrait<'a> {
    const FOO: u32;
}
struct MyStruct {}
impl MyTrait<'static> for MyStruct {
    const FOO: u32 = 99;
}

// EMIT_MIR user_type_annotations.match_assoc_const.built.after.mir
fn match_assoc_const() {
    match 8 {
        <MyStruct as MyTrait<'static>>::FOO => {}
        _ => {}
    }
}

// EMIT_MIR user_type_annotations.match_assoc_const_range.built.after.mir
fn match_assoc_const_range() {
    match 8 {
        ..<MyStruct as MyTrait<'static>>::FOO => {}
        <MyStruct as MyTrait<'static>>::FOO.. => {}
        _ => {}
    }
}
