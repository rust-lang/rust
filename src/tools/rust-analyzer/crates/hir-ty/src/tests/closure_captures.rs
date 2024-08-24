use base_db::salsa::InternKey;
use expect_test::{expect, Expect};
use hir_def::db::DefDatabase;
use itertools::Itertools;
use test_fixture::WithFixture;

use crate::db::{HirDatabase, InternedClosureId};
use crate::display::HirDisplay;
use crate::test_db::TestDB;

use super::visit_module;

fn check_closure_captures(ra_fixture: &str, expect: Expect) {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    let module = db.module_for_file(file_id);
    let def_map = module.def_map(&db);

    let mut defs = Vec::new();
    visit_module(&db, &def_map, module.local_id, &mut |it| defs.push(it));

    let mut captures_info = Vec::new();
    for def in defs {
        let infer = db.infer(def);
        let db = &db;
        captures_info.extend(infer.closure_info.iter().flat_map(|(closure_id, (captures, _))| {
            let closure = db.lookup_intern_closure(InternedClosureId::from_intern_id(closure_id.0));
            let (_, source_map) = db.body_with_source_map(closure.0);
            let closure_text_range = source_map
                .expr_syntax(closure.1)
                .expect("failed to map closure to SyntaxNode")
                .value
                .text_range();
            captures.iter().flat_map(move |capture| {
                // FIXME: Deduplicate this with hir::Local::sources().
                let (body, source_map) = db.body_with_source_map(closure.0);
                let local_text_ranges = match body.self_param.zip(source_map.self_param_syntax()) {
                    Some((param, source)) if param == capture.local() => {
                        vec![source.file_syntax(db).text_range()]
                    }
                    _ => source_map
                        .patterns_for_binding(capture.local())
                        .iter()
                        .map(|&definition| {
                            let src = source_map.pat_syntax(definition).unwrap();
                            src.file_syntax(db).text_range()
                        })
                        .collect(),
                };
                let place = capture.display_place(closure.0, db);
                let capture_ty = capture.ty.skip_binders().display_test(db).to_string();
                local_text_ranges.into_iter().map(move |local_text_range| {
                    (
                        closure_text_range,
                        local_text_range,
                        place.clone(),
                        capture_ty.clone(),
                        capture.kind(),
                    )
                })
            })
        }));
    }
    captures_info.sort_unstable_by_key(|(closure_text_range, local_text_range, ..)| {
        (closure_text_range.start(), local_text_range.start())
    });

    let rendered = captures_info
        .iter()
        .map(|(closure_text_range, local_text_range, place, capture_ty, capture_kind)| {
            format!(
                "{closure_text_range:?};{local_text_range:?} {capture_kind:?} {place} {capture_ty}"
            )
        })
        .join("\n");

    expect.assert_eq(&rendered);
}

#[test]
fn deref_in_let() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let a = &mut true;
    let closure = || { let b = *a; };
}
"#,
        expect!["53..71;0..75 ByRef(Shared) *a &'? bool"],
    );
}

#[test]
fn deref_then_ref_pattern() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let a = &mut true;
    let closure = || { let &mut ref b = a; };
}
"#,
        expect!["53..79;0..83 ByRef(Shared) *a &'? bool"],
    );
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let a = &mut true;
    let closure = || { let &mut ref mut b = a; };
}
"#,
        expect!["53..83;0..87 ByRef(Mut { kind: Default }) *a &'? mut bool"],
    );
}

#[test]
fn unique_borrow() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let a = &mut true;
    let closure = || { *a = false; };
}
"#,
        expect!["53..71;0..75 ByRef(Mut { kind: Default }) *a &'? mut bool"],
    );
}

#[test]
fn deref_ref_mut() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let a = &mut true;
    let closure = || { let ref mut b = *a; };
}
"#,
        expect!["53..79;0..83 ByRef(Mut { kind: Default }) *a &'? mut bool"],
    );
}

#[test]
fn let_else_not_consuming() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let a = &mut true;
    let closure = || { let _ = *a else { return; }; };
}
"#,
        expect!["53..88;0..92 ByRef(Shared) *a &'? bool"],
    );
}

#[test]
fn consume() {
    check_closure_captures(
        r#"
//- minicore:copy
struct NonCopy;
fn main() {
    let a = NonCopy;
    let closure = || { let b = a; };
}
"#,
        expect!["67..84;0..88 ByValue a NonCopy"],
    );
}

#[test]
fn ref_to_upvar() {
    check_closure_captures(
        r#"
//- minicore:copy
struct NonCopy;
fn main() {
    let mut a = NonCopy;
    let closure = || { let b = &a; };
    let closure = || { let c = &mut a; };
}
"#,
        expect![[r#"
            71..89;0..135 ByRef(Shared) a &'? NonCopy
            109..131;0..135 ByRef(Mut { kind: Default }) a &'? mut NonCopy"#]],
    );
}

#[test]
fn field() {
    check_closure_captures(
        r#"
//- minicore:copy
struct Foo { a: i32, b: i32 }
fn main() {
    let a = Foo { a: 0, b: 0 };
    let closure = || { let b = a.a; };
}
"#,
        expect!["92..111;0..115 ByRef(Shared) a.a &'? i32"],
    );
}

#[test]
fn fields_different_mode() {
    check_closure_captures(
        r#"
//- minicore:copy
struct NonCopy;
struct Foo { a: i32, b: i32, c: NonCopy, d: bool }
fn main() {
    let mut a = Foo { a: 0, b: 0 };
    let closure = || {
        let b = &a.a;
        let c = &mut a.b;
        let d = a.c;
    };
}
"#,
        expect![[r#"
            133..212;0..216 ByRef(Shared) a.a &'? i32
            133..212;0..216 ByRef(Mut { kind: Default }) a.b &'? mut i32
            133..212;0..216 ByValue a.c NonCopy"#]],
    );
}

#[test]
fn autoref() {
    check_closure_captures(
        r#"
//- minicore:copy
struct Foo;
impl Foo {
    fn imm(&self) {}
    fn mut_(&mut self) {}
}
fn main() {
    let mut a = Foo;
    let closure = || a.imm();
    let closure = || a.mut_();
}
"#,
        expect![[r#"
            123..133;0..168 ByRef(Shared) a &'? Foo
            153..164;0..168 ByRef(Mut { kind: Default }) a &'? mut Foo"#]],
    );
}

#[test]
fn captures_priority() {
    check_closure_captures(
        r#"
//- minicore:copy
struct NonCopy;
fn main() {
    let mut a = &mut true;
    // Max ByRef(Mut { kind: Default })
    let closure = || {
        *a = false;
        let b = &mut a;
    };
    // Max ByRef(Mut { kind: ClosureCapture })
    let closure = || {
        let b = *a;
        let c = &mut *a;
    };
    // Max ByValue
    let mut a = NonCopy;
    let closure = || {
        let b = a;
        let c = &mut a;
        let d = &a;
    };
}
"#,
        expect![[r#"
            113..167;0..430 ByRef(Mut { kind: Default }) a &'? mut &'? mut bool
            234..289;0..430 ByRef(Mut { kind: Default }) *a &'? mut bool
            353..426;0..430 ByValue a NonCopy"#]],
    );
}
