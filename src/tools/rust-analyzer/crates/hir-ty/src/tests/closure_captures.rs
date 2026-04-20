use expect_test::{Expect, expect};
use hir_def::{
    AdtId, DefWithBodyId, LocalFieldId, VariantId,
    expr_store::{Body, ExpressionStore},
    hir::{BindingId, ExprOrPatId},
};
use hir_expand::{HirFileId, files::InFileWrapper};
use itertools::Itertools;
use rustc_type_ir::inherent::{AdtDef as _, IntoKind};
use span::{Edition, TextRange};
use stdx::{format_to, never};
use syntax::{AstNode, AstPtr};
use test_fixture::WithFixture;

use crate::{
    InferenceResult,
    closure_analysis::Place,
    display::{DisplayTarget, HirDisplay},
    next_solver::TyKind,
    test_db::TestDB,
};

use super::{setup_tracing, visit_module};

fn display_place(db: &TestDB, store: &ExpressionStore, place: &Place, local: BindingId) -> String {
    let mut result = store[local].name.display(db, Edition::LATEST).to_string();
    let mut last_was_deref = false;
    for (i, proj) in place.projections.iter().enumerate() {
        match proj.kind {
            hir_ty::closure_analysis::ProjectionKind::Deref => {
                result.insert(0, '*');
                last_was_deref = true;
            }
            hir_ty::closure_analysis::ProjectionKind::Field { field_idx, variant_idx } => {
                if last_was_deref {
                    result.insert(0, '(');
                    result.push(')');
                    last_was_deref = false;
                }

                let ty = place.ty_before_projection(i);
                match ty.kind() {
                    TyKind::Tuple(_) => format_to!(result, ".{field_idx}"),
                    TyKind::Adt(adt_def, _) => {
                        let variant = match adt_def.def_id().0 {
                            AdtId::StructId(id) => VariantId::from(id),
                            AdtId::UnionId(id) => id.into(),
                            AdtId::EnumId(id) => {
                                // Can't really do that for an enum, unfortunately, so try to do something alike.
                                id.enum_variants(db).variants[variant_idx as usize].0.into()
                            }
                        };
                        let field = &variant.fields(db).fields()
                            [LocalFieldId::from_raw(la_arena::RawIdx::from_u32(field_idx))];
                        format_to!(result, ".{}", field.name.display(db, Edition::LATEST));
                    }
                    _ => never!("mismatching projection type"),
                }
            }
            _ => never!("unexpected projection kind"),
        }
    }
    result
}

fn check_closure_captures(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let _tracing = setup_tracing();
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    crate::attach_db(&db, || {
        let module = db.module_for_file(file_id.file_id(&db));
        let def_map = module.def_map(&db);

        let mut defs = Vec::new();
        visit_module(&db, def_map, module, &mut |it| defs.push(it));

        let mut captures_info = Vec::new();
        for def in defs {
            let def: DefWithBodyId = match def {
                hir_def::ModuleDefId::FunctionId(it) => it.into(),
                hir_def::ModuleDefId::EnumVariantId(it) => it.into(),
                hir_def::ModuleDefId::ConstId(it) => it.into(),
                hir_def::ModuleDefId::StaticId(it) => it.into(),
                _ => continue,
            };
            let infer = InferenceResult::of(&db, def);
            let db = &db;
            captures_info.extend(infer.closures_data.iter().flat_map(|(closure, closure_data)| {
                let (body, source_map) = Body::with_source_map(db, def);
                let closure_text_range = source_map
                    .expr_syntax(*closure)
                    .expect("failed to map closure to SyntaxNode")
                    .value
                    .text_range();
                closure_data.min_captures.values().flatten().map(move |capture| {
                    fn text_range<N: AstNode>(
                        db: &TestDB,
                        syntax: InFileWrapper<HirFileId, AstPtr<N>>,
                    ) -> TextRange {
                        let root = syntax.file_syntax(db);
                        syntax.value.to_node(&root).syntax().text_range()
                    }

                    // FIXME: Deduplicate this with hir::Local::sources().
                    let captured_local = capture.captured_local();
                    let local_text_range = match body.self_param.zip(source_map.self_param_syntax())
                    {
                        Some((param, source)) if param == captured_local => {
                            format!("{:?}", text_range(db, source))
                        }
                        _ => source_map
                            .patterns_for_binding(captured_local)
                            .iter()
                            .map(|&definition| {
                                text_range(db, source_map.pat_syntax(definition).unwrap())
                            })
                            .map(|it| format!("{it:?}"))
                            .join(", "),
                    };
                    let place = display_place(db, body, &capture.place, captured_local);
                    let capture_ty = capture
                        .captured_ty(db)
                        .display_test(db, DisplayTarget::from_crate(db, module.krate(db)))
                        .to_string();
                    let spans = capture
                        .info
                        .sources
                        .iter()
                        .flat_map(|span| match span.final_source() {
                            ExprOrPatId::ExprId(expr) => {
                                vec![text_range(db, source_map.expr_syntax(expr).unwrap())]
                            }
                            ExprOrPatId::PatId(pat) => {
                                vec![text_range(db, source_map.pat_syntax(pat).unwrap())]
                            }
                        })
                        .sorted_by_key(|it| it.start())
                        .map(|it| format!("{it:?}"))
                        .join(",");

                    (
                        closure_text_range,
                        local_text_range,
                        spans,
                        place,
                        capture_ty,
                        capture.info.capture_kind,
                    )
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
    })
}

#[test]
fn deref_in_let() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
fn main() {
    let a = &mut true;
    let closure = || { let b = *a; };
}
"#,
        expect!["53..71;20..21;66..68 ByRef(Immutable) *a &'<erased> bool"],
    );
}

#[test]
fn deref_then_ref_pattern() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
fn main() {
    let a = &mut true;
    let closure = || { let &mut ref b = a; };
}
"#,
        expect!["53..79;20..21;62..72 ByRef(Immutable) *a &'<erased> bool"],
    );
    check_closure_captures(
        r#"
//- minicore:copy, fn
fn main() {
    let a = &mut true;
    let closure = || { let &mut ref mut b = a; };
}
"#,
        expect!["53..83;20..21;62..76 ByRef(Mutable) *a &'<erased> mut bool"],
    );
}

#[test]
fn unique_borrow() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
fn main() {
    let a = &mut true;
    let closure = || { *a = false; };
}
"#,
        expect!["53..71;20..21;58..60 ByRef(Mutable) *a &'<erased> mut bool"],
    );
}

#[test]
fn deref_ref_mut() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
fn main() {
    let a = &mut true;
    let closure = || { let ref mut b = *a; };
}
"#,
        expect!["53..79;20..21;74..76 ByRef(Mutable) *a &'<erased> mut bool"],
    );
}

#[test]
fn let_else_not_consuming() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
fn main() {
    let a = &mut true;
    let closure = || { let _ = *a else { return; }; };
}
"#,
        expect![""],
    );
}

#[test]
fn consume() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
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
//- minicore:copy, fn
struct NonCopy;
fn main() {
    let mut a = NonCopy;
    let closure = || { let b = &a; };
    let closure = || { let c = &mut a; };
}
"#,
        expect![[r#"
            71..89;36..41;85..86 ByRef(Immutable) a &'<erased> NonCopy
            109..131;36..41;127..128 ByRef(Mutable) a &'<erased> mut NonCopy"#]],
    );
}

#[test]
fn field() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
struct Foo { a: i32, b: i32 }
fn main() {
    let a = Foo { a: 0, b: 0 };
    let closure = || { let b = a.a; };
}
"#,
        expect!["92..111;50..51;105..108 ByRef(Immutable) a.a &'<erased> i32"],
    );
}

#[test]
fn fields_different_mode() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
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
            133..212;87..92;155..158 ByRef(Immutable) a.a &'<erased> i32
            133..212;87..92;181..184 ByRef(Mutable) a.b &'<erased> mut i32
            133..212;87..92;202..205 ByValue a.c NonCopy"#]],
    );
}

#[test]
fn autoref() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
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
            123..133;92..97;126..127 ByRef(Immutable) a &'<erased> Foo
            153..164;92..97;156..157 ByRef(Mutable) a &'<erased> mut Foo"#]],
    );
}

#[test]
fn captures_priority() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
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
            113..167;36..41;127..128,159..160 ByRef(Mutable) a &'<erased> mut &'? mut bool
            231..304;196..201;252..253,276..277,296..297 ByValue a NonCopy"#]],
    );
}

#[test]
fn let_underscore() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
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
//- minicore:copy, fn
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
            125..163;36..41;134..135 ByRef(Immutable) a &'<erased> NonCopy
            183..225;36..41;192..193 ByRef(Mutable) a &'<erased> mut NonCopy"#]],
    );
}

#[test]
fn multiple_bindings() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
fn main() {
    let mut a = false;
    let mut closure = || { let (b | b) = a; };
}
"#,
        expect!["57..80;20..25;76..77 ByRef(Immutable) a &'<erased> bool"],
    );
}

#[test]
fn multiple_usages() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
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
            "57..149;20..25;79..80,99..100,123..124,134..135 ByRef(Mutable) a &'<erased> mut bool"
        ],
    );
}

#[test]
fn ref_then_deref() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
fn main() {
    let mut a = false;
    let mut closure = || { let b = *&mut a; };
}
"#,
        expect!["57..80;20..25;76..77 ByRef(Mutable) a &'<erased> mut bool"],
    );
}

#[test]
fn ref_of_ref() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
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
            54..72;20..25;68..69 ByRef(Immutable) a &'<erased> &'? bool
            92..114;20..25;110..111 ByRef(Mutable) a &'<erased> mut &'? bool
            158..176;124..125;172..173 ByRef(Immutable) a &'<erased> &'? mut bool
            196..218;124..125;214..215 ByRef(Mutable) a &'<erased> mut &'? mut bool"#]],
    );
}

#[test]
fn multiple_capture_usages() {
    check_closure_captures(
        r#"
//- minicore: copy, fn
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
        expect!["99..165;49..54;120..121,133..134 ByRef(Mutable) a &'<erased> mut A"],
    );
}

#[test]
fn let_binding_is_a_ref_capture_in_ref_binding() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
struct S;
fn main() {
    let mut s = S;
    let s_ref = &mut s;
    let mut s2 = S;
    let s_ref2 = &mut s2;
    let closure = || {
        if let ref cb = s_ref {
        } else if let ref mut cb = s_ref2 {
        }
    };
}
"#,
        expect![[r#"
            129..225;49..54;158..163 ByRef(Immutable) s_ref &'<erased> &'? mut S
            129..225;93..99;201..207 ByRef(Mutable) s_ref2 &'<erased> mut &'? mut S"#]],
    );
}

#[test]
fn let_binding_is_a_value_capture_in_binding() {
    check_closure_captures(
        r#"
//- minicore:copy, fn, option
struct Box(i32);
fn main() {
    let b = Some(Box(0));
    let closure = || {
        if let Some(b) = b {
            let _move = b;
        }
    };
}
"#,
        expect!["73..149;37..38;103..104 ByValue b Option<Box>"],
    );
}

#[test]
fn alias_needs_to_be_normalized() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
trait Trait {
    type Associated;
}
struct A;
struct B { x: i32 }
impl Trait for A {
    type Associated = B;
}
struct C { b: <A as Trait>::Associated }
fn main() {
    let c: C = C { b: B { x: 1 } };
    let closure = || {
        let _move = c.b.x;
    };
}
"#,
        expect!["220..257;174..175;245..250 ByRef(Immutable) c.b.x &'<erased> i32"],
    );
}

#[test]
fn nested_ref_captures_from_outer() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
fn f() {
    let a = 1;
    let a_closure = || {
        let b_closure = || {
            { a };
        };
    };
}
"#,
        expect![[r#"
            44..113;17..18;92..93 ByRef(Immutable) a &'<erased> i32
            73..106;17..18;92..93 ByRef(Immutable) a &'<erased> i32"#]],
    );
}

#[test]
fn nested_ref_captures() {
    check_closure_captures(
        r#"
//- minicore:copy, fn
fn f() {
    let a_closure = || {
        let b = 2;
        let b_closure = || {
            { b };
        };
    };
}
"#,
        expect!["77..110;46..47;96..97 ByRef(Immutable) b &'<erased> i32"],
    );
}
