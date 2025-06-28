use expect_test::{Expect, expect};
use hir_def::db::DefDatabase;
use hir_expand::{HirFileId, files::InFileWrapper};
use itertools::Itertools;
use salsa::plumbing::FromId;
use span::TextRange;
use syntax::{AstNode, AstPtr};
use test_fixture::WithFixture;

use crate::db::{HirDatabase, InternedClosureId};
use crate::display::{DisplayTarget, HirDisplay};
use crate::mir::MirSpan;
use crate::test_db::TestDB;

use super::visit_module;

fn check_closure_captures(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    let module = db.module_for_file(file_id.file_id(&db));
    let def_map = module.def_map(&db);

    let mut defs = Vec::new();
    visit_module(&db, def_map, module.local_id, &mut |it| defs.push(it));

    let mut captures_info = Vec::new();
    for def in defs {
        let def = match def {
            hir_def::ModuleDefId::FunctionId(it) => it.into(),
            hir_def::ModuleDefId::EnumVariantId(it) => it.into(),
            hir_def::ModuleDefId::ConstId(it) => it.into(),
            hir_def::ModuleDefId::StaticId(it) => it.into(),
            _ => continue,
        };
        let infer = db.infer(def);
        let db = &db;
        captures_info.extend(infer.closure_info.iter().flat_map(|(closure_id, (captures, _))| {
            let closure = db.lookup_intern_closure(InternedClosureId::from_id(closure_id.0));
            let source_map = db.body_with_source_map(closure.0).1;
            let closure_text_range = source_map
                .expr_syntax(closure.1)
                .expect("failed to map closure to SyntaxNode")
                .value
                .text_range();
            captures.iter().map(move |capture| {
                fn text_range<N: AstNode>(
                    db: &TestDB,
                    syntax: InFileWrapper<HirFileId, AstPtr<N>>,
                ) -> TextRange {
                    let root = syntax.file_syntax(db);
                    syntax.value.to_node(&root).syntax().text_range()
                }

                // FIXME: Deduplicate this with hir::Local::sources().
                let (body, source_map) = db.body_with_source_map(closure.0);
                let local_text_range = match body.self_param.zip(source_map.self_param_syntax()) {
                    Some((param, source)) if param == capture.local() => {
                        format!("{:?}", text_range(db, source))
                    }
                    _ => source_map
                        .patterns_for_binding(capture.local())
                        .iter()
                        .map(|&definition| {
                            text_range(db, source_map.pat_syntax(definition).unwrap())
                        })
                        .map(|it| format!("{it:?}"))
                        .join(", "),
                };
                let place = capture.display_place(closure.0, db);
                let capture_ty = capture
                    .ty
                    .skip_binders()
                    .display_test(db, DisplayTarget::from_crate(db, module.krate()))
                    .to_string();
                let spans = capture
                    .spans()
                    .iter()
                    .flat_map(|span| match *span {
                        MirSpan::ExprId(expr) => {
                            vec![text_range(db, source_map.expr_syntax(expr).unwrap())]
                        }
                        MirSpan::PatId(pat) => {
                            vec![text_range(db, source_map.pat_syntax(pat).unwrap())]
                        }
                        MirSpan::BindingId(binding) => source_map
                            .patterns_for_binding(binding)
                            .iter()
                            .map(|pat| text_range(db, source_map.pat_syntax(*pat).unwrap()))
                            .collect(),
                        MirSpan::SelfParam => {
                            vec![text_range(db, source_map.self_param_syntax().unwrap())]
                        }
                        MirSpan::Unknown => Vec::new(),
                    })
                    .sorted_by_key(|it| it.start())
                    .map(|it| format!("{it:?}"))
                    .join(",");

                (closure_text_range, local_text_range, spans, place, capture_ty, capture.kind())
            })
        }));
    }
    captures_info.sort_unstable_by_key(|(closure_text_range, local_text_range, ..)| {
        (closure_text_range.start(), local_text_range.clone())
    });

    let rendered = captures_info
        .iter()
        .map(|(closure_text_range, local_text_range, spans, place, capture_ty, capture_kind)| {
            format!(
                "{closure_text_range:?};{local_text_range};{spans} {capture_kind:?} {place} {capture_ty}"
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
        expect!["53..71;20..21;66..68 ByRef(Shared) *a &'? bool"],
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
        expect!["53..79;20..21;67..72 ByRef(Shared) *a &'? bool"],
    );
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let a = &mut true;
    let closure = || { let &mut ref mut b = a; };
}
"#,
        expect!["53..83;20..21;67..76 ByRef(Mut { kind: Default }) *a &'? mut bool"],
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
        expect!["53..71;20..21;58..60 ByRef(Mut { kind: Default }) *a &'? mut bool"],
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
        expect!["53..79;20..21;62..71 ByRef(Mut { kind: Default }) *a &'? mut bool"],
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
        expect!["53..88;20..21;66..68 ByRef(Shared) *a &'? bool"],
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
        expect!["67..84;36..37;80..81 ByValue a NonCopy"],
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
            71..89;36..41;84..86 ByRef(Shared) a &'? NonCopy
            109..131;36..41;122..128 ByRef(Mut { kind: Default }) a &'? mut NonCopy"#]],
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
        expect!["92..111;50..51;105..108 ByRef(Shared) a.a &'? i32"],
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
            133..212;87..92;154..158 ByRef(Shared) a.a &'? i32
            133..212;87..92;176..184 ByRef(Mut { kind: Default }) a.b &'? mut i32
            133..212;87..92;202..205 ByValue a.c NonCopy"#]],
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
            123..133;92..97;126..127 ByRef(Shared) a &'? Foo
            153..164;92..97;156..157 ByRef(Mut { kind: Default }) a &'? mut Foo"#]],
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
            113..167;36..41;127..128,154..160 ByRef(Mut { kind: Default }) a &'? mut &'? mut bool
            231..304;196..201;252..253,276..277,296..297 ByValue a NonCopy"#]],
    );
}

#[test]
fn let_underscore() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let mut a = true;
    let closure = || { let _ = a; };
}
"#,
        expect![""],
    );
}

#[test]
fn match_wildcard() {
    check_closure_captures(
        r#"
//- minicore:copy
struct NonCopy;
fn main() {
    let mut a = NonCopy;
    let closure = || match a {
        _ => {}
    };
    let closure = || match a {
        ref b => {}
    };
    let closure = || match a {
        ref mut b => {}
    };
}
"#,
        expect![[r#"
            125..163;36..41;134..135 ByRef(Shared) a &'? NonCopy
            183..225;36..41;192..193 ByRef(Mut { kind: Default }) a &'? mut NonCopy"#]],
    );
}

#[test]
fn multiple_bindings() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let mut a = false;
    let mut closure = || { let (b | b) = a; };
}
"#,
        expect!["57..80;20..25;76..77,76..77 ByRef(Shared) a &'? bool"],
    );
}

#[test]
fn multiple_usages() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let mut a = false;
    let mut closure = || {
        let b = &a;
        let c = &a;
        let d = &mut a;
        a = true;
    };
}
"#,
        expect![
            "57..149;20..25;78..80,98..100,118..124,134..135 ByRef(Mut { kind: Default }) a &'? mut bool"
        ],
    );
}

#[test]
fn ref_then_deref() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let mut a = false;
    let mut closure = || { let b = *&mut a; };
}
"#,
        expect!["57..80;20..25;71..77 ByRef(Mut { kind: Default }) a &'? mut bool"],
    );
}

#[test]
fn ref_of_ref() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let mut a = &false;
    let closure = || { let b = &a; };
    let closure = || { let b = &mut a; };
    let a = &mut false;
    let closure = || { let b = &a; };
    let closure = || { let b = &mut a; };
}
"#,
        expect![[r#"
            54..72;20..25;67..69 ByRef(Shared) a &'? &'? bool
            92..114;20..25;105..111 ByRef(Mut { kind: Default }) a &'? mut &'? bool
            158..176;124..125;171..173 ByRef(Shared) a &'? &'? mut bool
            196..218;124..125;209..215 ByRef(Mut { kind: Default }) a &'? mut &'? mut bool"#]],
    );
}

#[test]
fn multiple_capture_usages() {
    check_closure_captures(
        r#"
//- minicore:copy
struct A { a: i32, b: bool }
fn main() {
    let mut a = A { a: 123, b: false };
    let closure = |$0| {
        let b = a.b;
        a = A { a: 456, b: true };
    };
    closure();
}
"#,
        expect!["99..165;49..54;120..121,133..134 ByRef(Mut { kind: Default }) a &'? mut A"],
    );
}

#[test]
fn let_binding_is_a_ref_capture() {
    check_closure_captures(
        r#"
//- minicore:copy
struct S;
fn main() {
    let mut s = S;
    let s_ref = &mut s;
    let closure = || {
        if let ref cb = s_ref {
        }
    };
}
"#,
        expect!["83..135;49..54;112..117 ByRef(Shared) s_ref &'? &'? mut S"],
    );
}
