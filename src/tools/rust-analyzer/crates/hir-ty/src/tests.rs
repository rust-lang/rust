mod never_type;
mod coercion;
mod regression;
mod simple;
mod patterns;
mod traits;
mod method_resolution;
mod macros;
mod display_source_code;
mod incremental;
mod diagnostics;

use std::{collections::HashMap, env};

use base_db::{fixture::WithFixture, FileRange, SourceDatabaseExt};
use expect_test::Expect;
use hir_def::{
    body::{Body, BodySourceMap, SyntheticSyntax},
    db::{DefDatabase, InternDatabase},
    hir::{ExprId, Pat, PatId},
    item_scope::ItemScope,
    nameres::DefMap,
    src::HasSource,
    AssocItemId, DefWithBodyId, HasModule, LocalModuleId, Lookup, ModuleDefId,
};
use hir_expand::{db::ExpandDatabase, InFile};
use once_cell::race::OnceBool;
use stdx::format_to;
use syntax::{
    ast::{self, AstNode, HasName},
    SyntaxNode,
};
use tracing_subscriber::{layer::SubscriberExt, Registry};
use tracing_tree::HierarchicalLayer;
use triomphe::Arc;

use crate::{
    db::HirDatabase,
    display::HirDisplay,
    infer::{Adjustment, TypeMismatch},
    test_db::TestDB,
    InferenceResult, Ty,
};

// These tests compare the inference results for all expressions in a file
// against snapshots of the expected results using expect. Use
// `env UPDATE_EXPECT=1 cargo test -p hir_ty` to update the snapshots.

fn setup_tracing() -> Option<tracing::subscriber::DefaultGuard> {
    static ENABLE: OnceBool = OnceBool::new();
    if !ENABLE.get_or_init(|| env::var("CHALK_DEBUG").is_ok()) {
        return None;
    }

    let filter: tracing_subscriber::filter::Targets =
        env::var("CHALK_DEBUG").ok().and_then(|it| it.parse().ok()).unwrap_or_default();
    let layer = HierarchicalLayer::default()
        .with_indent_lines(true)
        .with_ansi(false)
        .with_indent_amount(2)
        .with_writer(std::io::stderr);
    let subscriber = Registry::default().with(filter).with(layer);
    Some(tracing::subscriber::set_default(subscriber))
}

#[track_caller]
fn check_types(ra_fixture: &str) {
    check_impl(ra_fixture, false, true, false)
}

#[track_caller]
fn check_types_source_code(ra_fixture: &str) {
    check_impl(ra_fixture, false, true, true)
}

#[track_caller]
fn check_no_mismatches(ra_fixture: &str) {
    check_impl(ra_fixture, true, false, false)
}

#[track_caller]
fn check(ra_fixture: &str) {
    check_impl(ra_fixture, false, false, false)
}

#[track_caller]
fn check_impl(ra_fixture: &str, allow_none: bool, only_types: bool, display_source: bool) {
    let _tracing = setup_tracing();
    let (db, files) = TestDB::with_many_files(ra_fixture);

    let mut had_annotations = false;
    let mut mismatches = HashMap::new();
    let mut types = HashMap::new();
    let mut adjustments = HashMap::<_, Vec<_>>::new();
    for (file_id, annotations) in db.extract_annotations() {
        for (range, expected) in annotations {
            let file_range = FileRange { file_id, range };
            if only_types {
                types.insert(file_range, expected);
            } else if expected.starts_with("type: ") {
                types.insert(file_range, expected.trim_start_matches("type: ").to_string());
            } else if expected.starts_with("expected") {
                mismatches.insert(file_range, expected);
            } else if expected.starts_with("adjustments:") {
                adjustments.insert(
                    file_range,
                    expected
                        .trim_start_matches("adjustments:")
                        .trim()
                        .split(',')
                        .map(|it| it.trim().to_string())
                        .filter(|it| !it.is_empty())
                        .collect(),
                );
            } else {
                panic!("unexpected annotation: {expected}");
            }
            had_annotations = true;
        }
    }
    assert!(had_annotations || allow_none, "no `//^` annotations found");

    let mut defs: Vec<DefWithBodyId> = Vec::new();
    for file_id in files {
        let module = db.module_for_file_opt(file_id);
        let module = match module {
            Some(m) => m,
            None => continue,
        };
        let def_map = module.def_map(&db);
        visit_module(&db, &def_map, module.local_id, &mut |it| defs.push(it));
    }
    defs.sort_by_key(|def| match def {
        DefWithBodyId::FunctionId(it) => {
            let loc = it.lookup(&db);
            loc.source(&db).value.syntax().text_range().start()
        }
        DefWithBodyId::ConstId(it) => {
            let loc = it.lookup(&db);
            loc.source(&db).value.syntax().text_range().start()
        }
        DefWithBodyId::StaticId(it) => {
            let loc = it.lookup(&db);
            loc.source(&db).value.syntax().text_range().start()
        }
        DefWithBodyId::VariantId(it) => {
            let loc = db.lookup_intern_enum(it.parent);
            loc.source(&db).value.syntax().text_range().start()
        }
        DefWithBodyId::InTypeConstId(it) => it.source(&db).syntax().text_range().start(),
    });
    let mut unexpected_type_mismatches = String::new();
    for def in defs {
        let (body, body_source_map) = db.body_with_source_map(def);
        let inference_result = db.infer(def);

        for (pat, mut ty) in inference_result.type_of_pat.iter() {
            if let Pat::Bind { id, .. } = body.pats[pat] {
                ty = &inference_result.type_of_binding[id];
            }
            let node = match pat_node(&body_source_map, pat, &db) {
                Some(value) => value,
                None => continue,
            };
            let range = node.as_ref().original_file_range(&db);
            if let Some(expected) = types.remove(&range) {
                let actual = if display_source {
                    ty.display_source_code(&db, def.module(&db), true).unwrap()
                } else {
                    ty.display_test(&db).to_string()
                };
                assert_eq!(actual, expected, "type annotation differs at {:#?}", range.range);
            }
        }

        for (expr, ty) in inference_result.type_of_expr.iter() {
            let node = match expr_node(&body_source_map, expr, &db) {
                Some(value) => value,
                None => continue,
            };
            let range = node.as_ref().original_file_range(&db);
            if let Some(expected) = types.remove(&range) {
                let actual = if display_source {
                    ty.display_source_code(&db, def.module(&db), true).unwrap()
                } else {
                    ty.display_test(&db).to_string()
                };
                assert_eq!(actual, expected, "type annotation differs at {:#?}", range.range);
            }
            if let Some(expected) = adjustments.remove(&range) {
                let adjustments = inference_result
                    .expr_adjustments
                    .get(&expr)
                    .map_or_else(Default::default, |it| &**it);
                assert_eq!(
                    expected,
                    adjustments
                        .iter()
                        .map(|Adjustment { kind, .. }| format!("{kind:?}"))
                        .collect::<Vec<_>>()
                );
            }
        }

        for (expr_or_pat, mismatch) in inference_result.type_mismatches() {
            let Some(node) = (match expr_or_pat {
                hir_def::hir::ExprOrPatId::ExprId(expr) => expr_node(&body_source_map, expr, &db),
                hir_def::hir::ExprOrPatId::PatId(pat) => pat_node(&body_source_map, pat, &db),
            }) else {
                continue;
            };
            let range = node.as_ref().original_file_range(&db);
            let actual = format!(
                "expected {}, got {}",
                mismatch.expected.display_test(&db),
                mismatch.actual.display_test(&db)
            );
            match mismatches.remove(&range) {
                Some(annotation) => assert_eq!(actual, annotation),
                None => format_to!(unexpected_type_mismatches, "{:?}: {}\n", range.range, actual),
            }
        }
    }

    let mut buf = String::new();
    if !unexpected_type_mismatches.is_empty() {
        format_to!(buf, "Unexpected type mismatches:\n{}", unexpected_type_mismatches);
    }
    if !mismatches.is_empty() {
        format_to!(buf, "Unchecked mismatch annotations:\n");
        for m in mismatches {
            format_to!(buf, "{:?}: {}\n", m.0.range, m.1);
        }
    }
    if !types.is_empty() {
        format_to!(buf, "Unchecked type annotations:\n");
        for t in types {
            format_to!(buf, "{:?}: type {}\n", t.0.range, t.1);
        }
    }
    if !adjustments.is_empty() {
        format_to!(buf, "Unchecked adjustments annotations:\n");
        for t in adjustments {
            format_to!(buf, "{:?}: type {:?}\n", t.0.range, t.1);
        }
    }
    assert!(buf.is_empty(), "{}", buf);
}

fn expr_node(
    body_source_map: &BodySourceMap,
    expr: ExprId,
    db: &TestDB,
) -> Option<InFile<SyntaxNode>> {
    Some(match body_source_map.expr_syntax(expr) {
        Ok(sp) => {
            let root = db.parse_or_expand(sp.file_id);
            sp.map(|ptr| ptr.to_node(&root).syntax().clone())
        }
        Err(SyntheticSyntax) => return None,
    })
}

fn pat_node(
    body_source_map: &BodySourceMap,
    pat: PatId,
    db: &TestDB,
) -> Option<InFile<SyntaxNode>> {
    Some(match body_source_map.pat_syntax(pat) {
        Ok(sp) => {
            let root = db.parse_or_expand(sp.file_id);
            sp.map(|ptr| {
                ptr.either(
                    |it| it.to_node(&root).syntax().clone(),
                    |it| it.to_node(&root).syntax().clone(),
                )
            })
        }
        Err(SyntheticSyntax) => return None,
    })
}

fn infer(ra_fixture: &str) -> String {
    infer_with_mismatches(ra_fixture, false)
}

fn infer_with_mismatches(content: &str, include_mismatches: bool) -> String {
    let _tracing = setup_tracing();
    let (db, file_id) = TestDB::with_single_file(content);

    let mut buf = String::new();

    let mut infer_def = |inference_result: Arc<InferenceResult>,
                         body: Arc<Body>,
                         body_source_map: Arc<BodySourceMap>| {
        let mut types: Vec<(InFile<SyntaxNode>, &Ty)> = Vec::new();
        let mut mismatches: Vec<(InFile<SyntaxNode>, &TypeMismatch)> = Vec::new();

        for (pat, mut ty) in inference_result.type_of_pat.iter() {
            if let Pat::Bind { id, .. } = body.pats[pat] {
                ty = &inference_result.type_of_binding[id];
            }
            let syntax_ptr = match body_source_map.pat_syntax(pat) {
                Ok(sp) => {
                    let root = db.parse_or_expand(sp.file_id);
                    sp.map(|ptr| {
                        ptr.either(
                            |it| it.to_node(&root).syntax().clone(),
                            |it| it.to_node(&root).syntax().clone(),
                        )
                    })
                }
                Err(SyntheticSyntax) => continue,
            };
            types.push((syntax_ptr.clone(), ty));
            if let Some(mismatch) = inference_result.type_mismatch_for_pat(pat) {
                mismatches.push((syntax_ptr, mismatch));
            }
        }

        for (expr, ty) in inference_result.type_of_expr.iter() {
            let node = match body_source_map.expr_syntax(expr) {
                Ok(sp) => {
                    let root = db.parse_or_expand(sp.file_id);
                    sp.map(|ptr| ptr.to_node(&root).syntax().clone())
                }
                Err(SyntheticSyntax) => continue,
            };
            types.push((node.clone(), ty));
            if let Some(mismatch) = inference_result.type_mismatch_for_expr(expr) {
                mismatches.push((node, mismatch));
            }
        }

        // sort ranges for consistency
        types.sort_by_key(|(node, _)| {
            let range = node.value.text_range();
            (range.start(), range.end())
        });
        for (node, ty) in &types {
            let (range, text) = if let Some(self_param) = ast::SelfParam::cast(node.value.clone()) {
                (self_param.name().unwrap().syntax().text_range(), "self".to_string())
            } else {
                (node.value.text_range(), node.value.text().to_string().replace('\n', " "))
            };
            let macro_prefix = if node.file_id != file_id.into() { "!" } else { "" };
            format_to!(
                buf,
                "{}{:?} '{}': {}\n",
                macro_prefix,
                range,
                ellipsize(text, 15),
                ty.display_test(&db)
            );
        }
        if include_mismatches {
            mismatches.sort_by_key(|(node, _)| {
                let range = node.value.text_range();
                (range.start(), range.end())
            });
            for (src_ptr, mismatch) in &mismatches {
                let range = src_ptr.value.text_range();
                let macro_prefix = if src_ptr.file_id != file_id.into() { "!" } else { "" };
                format_to!(
                    buf,
                    "{}{:?}: expected {}, got {}\n",
                    macro_prefix,
                    range,
                    mismatch.expected.display_test(&db),
                    mismatch.actual.display_test(&db),
                );
            }
        }
    };

    let module = db.module_for_file(file_id);
    let def_map = module.def_map(&db);

    let mut defs: Vec<DefWithBodyId> = Vec::new();
    visit_module(&db, &def_map, module.local_id, &mut |it| defs.push(it));
    defs.sort_by_key(|def| match def {
        DefWithBodyId::FunctionId(it) => {
            let loc = it.lookup(&db);
            loc.source(&db).value.syntax().text_range().start()
        }
        DefWithBodyId::ConstId(it) => {
            let loc = it.lookup(&db);
            loc.source(&db).value.syntax().text_range().start()
        }
        DefWithBodyId::StaticId(it) => {
            let loc = it.lookup(&db);
            loc.source(&db).value.syntax().text_range().start()
        }
        DefWithBodyId::VariantId(it) => {
            let loc = db.lookup_intern_enum(it.parent);
            loc.source(&db).value.syntax().text_range().start()
        }
        DefWithBodyId::InTypeConstId(it) => it.source(&db).syntax().text_range().start(),
    });
    for def in defs {
        let (body, source_map) = db.body_with_source_map(def);
        let infer = db.infer(def);
        infer_def(infer, body, source_map);
    }

    buf.truncate(buf.trim_end().len());
    buf
}

fn visit_module(
    db: &TestDB,
    crate_def_map: &DefMap,
    module_id: LocalModuleId,
    cb: &mut dyn FnMut(DefWithBodyId),
) {
    visit_scope(db, crate_def_map, &crate_def_map[module_id].scope, cb);
    for impl_id in crate_def_map[module_id].scope.impls() {
        let impl_data = db.impl_data(impl_id);
        for &item in impl_data.items.iter() {
            match item {
                AssocItemId::FunctionId(it) => {
                    let def = it.into();
                    cb(def);
                    let body = db.body(def);
                    visit_body(db, &body, cb);
                }
                AssocItemId::ConstId(it) => {
                    let def = it.into();
                    cb(def);
                    let body = db.body(def);
                    visit_body(db, &body, cb);
                }
                AssocItemId::TypeAliasId(_) => (),
            }
        }
    }

    fn visit_scope(
        db: &TestDB,
        crate_def_map: &DefMap,
        scope: &ItemScope,
        cb: &mut dyn FnMut(DefWithBodyId),
    ) {
        for decl in scope.declarations() {
            match decl {
                ModuleDefId::FunctionId(it) => {
                    let def = it.into();
                    cb(def);
                    let body = db.body(def);
                    visit_body(db, &body, cb);
                }
                ModuleDefId::ConstId(it) => {
                    let def = it.into();
                    cb(def);
                    let body = db.body(def);
                    visit_body(db, &body, cb);
                }
                ModuleDefId::StaticId(it) => {
                    let def = it.into();
                    cb(def);
                    let body = db.body(def);
                    visit_body(db, &body, cb);
                }
                ModuleDefId::AdtId(hir_def::AdtId::EnumId(it)) => {
                    db.enum_data(it)
                        .variants
                        .iter()
                        .map(|(id, _)| hir_def::EnumVariantId { parent: it, local_id: id })
                        .for_each(|it| {
                            let def = it.into();
                            cb(def);
                            let body = db.body(def);
                            visit_body(db, &body, cb);
                        });
                }
                ModuleDefId::TraitId(it) => {
                    let trait_data = db.trait_data(it);
                    for &(_, item) in trait_data.items.iter() {
                        match item {
                            AssocItemId::FunctionId(it) => cb(it.into()),
                            AssocItemId::ConstId(it) => cb(it.into()),
                            AssocItemId::TypeAliasId(_) => (),
                        }
                    }
                }
                ModuleDefId::ModuleId(it) => visit_module(db, crate_def_map, it.local_id, cb),
                _ => (),
            }
        }
    }

    fn visit_body(db: &TestDB, body: &Body, cb: &mut dyn FnMut(DefWithBodyId)) {
        for (_, def_map) in body.blocks(db) {
            for (mod_id, _) in def_map.modules() {
                visit_module(db, &def_map, mod_id, cb);
            }
        }
    }
}

fn ellipsize(mut text: String, max_len: usize) -> String {
    if text.len() <= max_len {
        return text;
    }
    let ellipsis = "...";
    let e_len = ellipsis.len();
    let mut prefix_len = (max_len - e_len) / 2;
    while !text.is_char_boundary(prefix_len) {
        prefix_len += 1;
    }
    let mut suffix_len = max_len - e_len - prefix_len;
    while !text.is_char_boundary(text.len() - suffix_len) {
        suffix_len += 1;
    }
    text.replace_range(prefix_len..text.len() - suffix_len, ellipsis);
    text
}

fn check_infer(ra_fixture: &str, expect: Expect) {
    let mut actual = infer(ra_fixture);
    actual.push('\n');
    expect.assert_eq(&actual);
}

fn check_infer_with_mismatches(ra_fixture: &str, expect: Expect) {
    let mut actual = infer_with_mismatches(ra_fixture, true);
    actual.push('\n');
    expect.assert_eq(&actual);
}

#[test]
fn salsa_bug() {
    let (mut db, pos) = TestDB::with_position(
        "
        //- /lib.rs
        trait Index {
            type Output;
        }

        type Key<S: UnificationStoreBase> = <S as UnificationStoreBase>::Key;

        pub trait UnificationStoreBase: Index<Output = Key<Self>> {
            type Key;

            fn len(&self) -> usize;
        }

        pub trait UnificationStoreMut: UnificationStoreBase {
            fn push(&mut self, value: Self::Key);
        }

        fn main() {
            let x = 1;
            x.push(1);$0
        }
    ",
    );

    let module = db.module_for_file(pos.file_id);
    let crate_def_map = module.def_map(&db);
    visit_module(&db, &crate_def_map, module.local_id, &mut |def| {
        db.infer(def);
    });

    let new_text = "
        //- /lib.rs
        trait Index {
            type Output;
        }

        type Key<S: UnificationStoreBase> = <S as UnificationStoreBase>::Key;

        pub trait UnificationStoreBase: Index<Output = Key<Self>> {
            type Key;

            fn len(&self) -> usize;
        }

        pub trait UnificationStoreMut: UnificationStoreBase {
            fn push(&mut self, value: Self::Key);
        }

        fn main() {

            let x = 1;
            x.push(1);
        }
    ";

    db.set_file_text(pos.file_id, Arc::from(new_text));

    let module = db.module_for_file(pos.file_id);
    let crate_def_map = module.def_map(&db);
    visit_module(&db, &crate_def_map, module.local_id, &mut |def| {
        db.infer(def);
    });
}
