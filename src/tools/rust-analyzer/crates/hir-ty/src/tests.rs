mod closure_captures;
mod coercion;
mod diagnostics;
mod display_source_code;
mod incremental;
mod macros;
mod method_resolution;
mod never_type;
mod patterns;
mod regression;
mod simple;
mod traits;
mod type_alias_impl_traits;

use std::env;
use std::sync::LazyLock;

use base_db::{Crate, SourceDatabase};
use expect_test::Expect;
use hir_def::{
    AssocItemId, DefWithBodyId, HasModule, LocalModuleId, Lookup, ModuleDefId, SyntheticSyntax,
    db::DefDatabase,
    expr_store::{Body, BodySourceMap},
    hir::{ExprId, Pat, PatId},
    item_scope::ItemScope,
    nameres::DefMap,
    src::HasSource,
};
use hir_expand::{FileRange, InFile, db::ExpandDatabase};
use itertools::Itertools;
use rustc_hash::FxHashMap;
use stdx::format_to;
use syntax::{
    SyntaxNode,
    ast::{self, AstNode, HasName},
};
use test_fixture::WithFixture;
use tracing_subscriber::{Registry, layer::SubscriberExt};
use tracing_tree::HierarchicalLayer;
use triomphe::Arc;

use crate::{
    InferenceResult, Ty,
    db::HirDatabase,
    display::{DisplayTarget, HirDisplay},
    infer::{Adjustment, TypeMismatch},
    test_db::TestDB,
};

// These tests compare the inference results for all expressions in a file
// against snapshots of the expected results using expect. Use
// `env UPDATE_EXPECT=1 cargo test -p hir_ty` to update the snapshots.

fn setup_tracing() -> Option<tracing::subscriber::DefaultGuard> {
    static ENABLE: LazyLock<bool> = LazyLock::new(|| env::var("CHALK_DEBUG").is_ok());
    if !*ENABLE {
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
fn check_types(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    check_impl(ra_fixture, false, true, false)
}

#[track_caller]
fn check_types_source_code(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    check_impl(ra_fixture, false, true, true)
}

#[track_caller]
fn check_no_mismatches(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    check_impl(ra_fixture, true, false, false)
}

#[track_caller]
fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    check_impl(ra_fixture, false, false, false)
}

#[track_caller]
fn check_impl(
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    allow_none: bool,
    only_types: bool,
    display_source: bool,
) {
    let _tracing = setup_tracing();
    let (db, files) = TestDB::with_many_files(ra_fixture);

    let mut had_annotations = false;
    let mut mismatches = FxHashMap::default();
    let mut types = FxHashMap::default();
    let mut adjustments = FxHashMap::default();
    for (file_id, annotations) in db.extract_annotations() {
        for (range, expected) in annotations {
            let file_range = FileRange { file_id, range };
            if only_types {
                types.insert(file_range, expected);
            } else if expected.starts_with("type: ") {
                types.insert(file_range, expected.trim_start_matches("type: ").to_owned());
            } else if expected.starts_with("expected") {
                mismatches.insert(file_range, expected);
            } else if expected.starts_with("adjustments:") {
                adjustments.insert(
                    file_range,
                    expected.trim_start_matches("adjustments:").trim().to_owned(),
                );
            } else {
                panic!("unexpected annotation: {expected} @ {range:?}");
            }
            had_annotations = true;
        }
    }
    assert!(had_annotations || allow_none, "no `//^` annotations found");

    let mut defs: Vec<(DefWithBodyId, Crate)> = Vec::new();
    for file_id in files {
        let module = db.module_for_file_opt(file_id.file_id(&db));
        let module = match module {
            Some(m) => m,
            None => continue,
        };
        let def_map = module.def_map(&db);
        visit_module(&db, def_map, module.local_id, &mut |it| {
            let def = match it {
                ModuleDefId::FunctionId(it) => it.into(),
                ModuleDefId::EnumVariantId(it) => it.into(),
                ModuleDefId::ConstId(it) => it.into(),
                ModuleDefId::StaticId(it) => it.into(),
                _ => return,
            };
            defs.push((def, module.krate()))
        });
    }
    defs.sort_by_key(|(def, _)| match def {
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
            let loc = it.lookup(&db);
            loc.source(&db).value.syntax().text_range().start()
        }
    });
    let mut unexpected_type_mismatches = String::new();
    for (def, krate) in defs {
        let display_target = DisplayTarget::from_crate(&db, krate);
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
            let range = node.as_ref().original_file_range_rooted(&db);
            if let Some(expected) = types.remove(&range) {
                let actual = if display_source {
                    ty.display_source_code(&db, def.module(&db), true).unwrap()
                } else {
                    ty.display_test(&db, display_target).to_string()
                };
                assert_eq!(actual, expected, "type annotation differs at {:#?}", range.range);
            }
        }

        for (expr, ty) in inference_result.type_of_expr.iter() {
            let node = match expr_node(&body_source_map, expr, &db) {
                Some(value) => value,
                None => continue,
            };
            let range = node.as_ref().original_file_range_rooted(&db);
            if let Some(expected) = types.remove(&range) {
                let actual = if display_source {
                    ty.display_source_code(&db, def.module(&db), true).unwrap()
                } else {
                    ty.display_test(&db, display_target).to_string()
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
                        .join(", ")
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
            let range = node.as_ref().original_file_range_rooted(&db);
            let actual = format!(
                "expected {}, got {}",
                mismatch.expected.display_test(&db, display_target),
                mismatch.actual.display_test(&db, display_target)
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
            sp.map(|ptr| ptr.to_node(&root).syntax().clone())
        }
        Err(SyntheticSyntax) => return None,
    })
}

fn infer(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> String {
    infer_with_mismatches(ra_fixture, false)
}

fn infer_with_mismatches(content: &str, include_mismatches: bool) -> String {
    let _tracing = setup_tracing();
    let (db, file_id) = TestDB::with_single_file(content);

    let mut buf = String::new();

    let mut infer_def = |inference_result: Arc<InferenceResult>,
                         body: Arc<Body>,
                         body_source_map: Arc<BodySourceMap>,
                         krate: Crate| {
        let display_target = DisplayTarget::from_crate(&db, krate);
        let mut types: Vec<(InFile<SyntaxNode>, &Ty)> = Vec::new();
        let mut mismatches: Vec<(InFile<SyntaxNode>, &TypeMismatch)> = Vec::new();

        if let Some(self_param) = body.self_param {
            let ty = &inference_result.type_of_binding[self_param];
            if let Some(syntax_ptr) = body_source_map.self_param_syntax() {
                let root = db.parse_or_expand(syntax_ptr.file_id);
                let node = syntax_ptr.map(|ptr| ptr.to_node(&root).syntax().clone());
                types.push((node, ty));
            }
        }

        for (pat, mut ty) in inference_result.type_of_pat.iter() {
            if let Pat::Bind { id, .. } = body.pats[pat] {
                ty = &inference_result.type_of_binding[id];
            }
            let node = match body_source_map.pat_syntax(pat) {
                Ok(sp) => {
                    let root = db.parse_or_expand(sp.file_id);
                    sp.map(|ptr| ptr.to_node(&root).syntax().clone())
                }
                Err(SyntheticSyntax) => continue,
            };
            types.push((node.clone(), ty));
            if let Some(mismatch) = inference_result.type_mismatch_for_pat(pat) {
                mismatches.push((node, mismatch));
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
                (self_param.name().unwrap().syntax().text_range(), "self".to_owned())
            } else {
                (node.value.text_range(), node.value.text().to_string().replace('\n', " "))
            };
            let macro_prefix = if node.file_id != file_id { "!" } else { "" };
            format_to!(
                buf,
                "{}{:?} '{}': {}\n",
                macro_prefix,
                range,
                ellipsize(text, 15),
                ty.display_test(&db, display_target)
            );
        }
        if include_mismatches {
            mismatches.sort_by_key(|(node, _)| {
                let range = node.value.text_range();
                (range.start(), range.end())
            });
            for (src_ptr, mismatch) in &mismatches {
                let range = src_ptr.value.text_range();
                let macro_prefix = if src_ptr.file_id != file_id { "!" } else { "" };
                format_to!(
                    buf,
                    "{}{:?}: expected {}, got {}\n",
                    macro_prefix,
                    range,
                    mismatch.expected.display_test(&db, display_target),
                    mismatch.actual.display_test(&db, display_target),
                );
            }
        }
    };

    let module = db.module_for_file(file_id.file_id(&db));
    let def_map = module.def_map(&db);

    let mut defs: Vec<(DefWithBodyId, Crate)> = Vec::new();
    visit_module(&db, def_map, module.local_id, &mut |it| {
        let def = match it {
            ModuleDefId::FunctionId(it) => it.into(),
            ModuleDefId::EnumVariantId(it) => it.into(),
            ModuleDefId::ConstId(it) => it.into(),
            ModuleDefId::StaticId(it) => it.into(),
            _ => return,
        };
        defs.push((def, module.krate()))
    });
    defs.sort_by_key(|(def, _)| match def {
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
            let loc = it.lookup(&db);
            loc.source(&db).value.syntax().text_range().start()
        }
    });
    for (def, krate) in defs {
        let (body, source_map) = db.body_with_source_map(def);
        let infer = db.infer(def);
        infer_def(infer, body, source_map, krate);
    }

    buf.truncate(buf.trim_end().len());
    buf
}

pub(crate) fn visit_module(
    db: &TestDB,
    crate_def_map: &DefMap,
    module_id: LocalModuleId,
    cb: &mut dyn FnMut(ModuleDefId),
) {
    visit_scope(db, crate_def_map, &crate_def_map[module_id].scope, cb);
    for impl_id in crate_def_map[module_id].scope.impls() {
        let impl_data = impl_id.impl_items(db);
        for &(_, item) in impl_data.items.iter() {
            match item {
                AssocItemId::FunctionId(it) => {
                    let body = db.body(it.into());
                    cb(it.into());
                    visit_body(db, &body, cb);
                }
                AssocItemId::ConstId(it) => {
                    let body = db.body(it.into());
                    cb(it.into());
                    visit_body(db, &body, cb);
                }
                AssocItemId::TypeAliasId(it) => {
                    cb(it.into());
                }
            }
        }
    }

    fn visit_scope(
        db: &TestDB,
        crate_def_map: &DefMap,
        scope: &ItemScope,
        cb: &mut dyn FnMut(ModuleDefId),
    ) {
        for decl in scope.declarations() {
            cb(decl);
            match decl {
                ModuleDefId::FunctionId(it) => {
                    let body = db.body(it.into());
                    visit_body(db, &body, cb);
                }
                ModuleDefId::ConstId(it) => {
                    let body = db.body(it.into());
                    visit_body(db, &body, cb);
                }
                ModuleDefId::StaticId(it) => {
                    let body = db.body(it.into());
                    visit_body(db, &body, cb);
                }
                ModuleDefId::AdtId(hir_def::AdtId::EnumId(it)) => {
                    it.enum_variants(db).variants.iter().for_each(|&(it, _, _)| {
                        let body = db.body(it.into());
                        cb(it.into());
                        visit_body(db, &body, cb);
                    });
                }
                ModuleDefId::TraitId(it) => {
                    let trait_data = it.trait_items(db);
                    for &(_, item) in trait_data.items.iter() {
                        match item {
                            AssocItemId::FunctionId(it) => cb(it.into()),
                            AssocItemId::ConstId(it) => cb(it.into()),
                            AssocItemId::TypeAliasId(it) => cb(it.into()),
                        }
                    }
                }
                ModuleDefId::ModuleId(it) => visit_module(db, crate_def_map, it.local_id, cb),
                _ => (),
            }
        }
    }

    fn visit_body(db: &TestDB, body: &Body, cb: &mut dyn FnMut(ModuleDefId)) {
        for (_, def_map) in body.blocks(db) {
            for (mod_id, _) in def_map.modules() {
                visit_module(db, def_map, mod_id, cb);
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

fn check_infer(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let mut actual = infer(ra_fixture);
    actual.push('\n');
    expect.assert_eq(&actual);
}

fn check_infer_with_mismatches(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
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

    let module = db.module_for_file(pos.file_id.file_id(&db));
    let crate_def_map = module.def_map(&db);
    visit_module(&db, crate_def_map, module.local_id, &mut |def| {
        db.infer(match def {
            ModuleDefId::FunctionId(it) => it.into(),
            ModuleDefId::EnumVariantId(it) => it.into(),
            ModuleDefId::ConstId(it) => it.into(),
            ModuleDefId::StaticId(it) => it.into(),
            _ => return,
        });
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

    db.set_file_text(pos.file_id.file_id(&db), new_text);

    let module = db.module_for_file(pos.file_id.file_id(&db));
    let crate_def_map = module.def_map(&db);
    visit_module(&db, crate_def_map, module.local_id, &mut |def| {
        db.infer(match def {
            ModuleDefId::FunctionId(it) => it.into(),
            ModuleDefId::EnumVariantId(it) => it.into(),
            ModuleDefId::ConstId(it) => it.into(),
            ModuleDefId::StaticId(it) => it.into(),
            _ => return,
        });
    });
}
