mod never_type;
mod coercion;
mod regression;
mod simple;
mod patterns;
mod traits;
mod method_resolution;
mod macros;
mod display_source_code;

use std::{env, sync::Arc};

use base_db::{fixture::WithFixture, FileRange, SourceDatabase, SourceDatabaseExt};
use expect_test::Expect;
use hir_def::{
    body::{BodySourceMap, SyntheticSyntax},
    child_by_source::ChildBySource,
    db::DefDatabase,
    item_scope::ItemScope,
    keys,
    nameres::DefMap,
    AssocItemId, DefWithBodyId, LocalModuleId, Lookup, ModuleDefId,
};
use hir_expand::{db::AstDatabase, InFile};
use once_cell::race::OnceBool;
use stdx::format_to;
use syntax::{
    algo,
    ast::{self, AstNode, NameOwner},
    SyntaxNode,
};
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};
use tracing_tree::HierarchicalLayer;

use crate::{
    db::HirDatabase, display::HirDisplay, infer::TypeMismatch, test_db::TestDB, InferenceResult, Ty,
};

// These tests compare the inference results for all expressions in a file
// against snapshots of the expected results using expect. Use
// `env UPDATE_EXPECT=1 cargo test -p hir_ty` to update the snapshots.

fn setup_tracing() -> Option<tracing::subscriber::DefaultGuard> {
    static ENABLE: OnceBool = OnceBool::new();
    if !ENABLE.get_or_init(|| env::var("CHALK_DEBUG").is_ok()) {
        return None;
    }

    let filter = EnvFilter::from_env("CHALK_DEBUG");
    let layer = HierarchicalLayer::default()
        .with_indent_lines(true)
        .with_ansi(false)
        .with_indent_amount(2)
        .with_writer(std::io::stderr);
    let subscriber = Registry::default().with(filter).with(layer);
    Some(tracing::subscriber::set_default(subscriber))
}

fn check_types(ra_fixture: &str) {
    check_types_impl(ra_fixture, false)
}

fn check_types_source_code(ra_fixture: &str) {
    check_types_impl(ra_fixture, true)
}

fn check_types_impl(ra_fixture: &str, display_source: bool) {
    let _tracing = setup_tracing();
    let db = TestDB::with_files(ra_fixture);
    let mut checked_one = false;
    for (file_id, annotations) in db.extract_annotations() {
        for (range, expected) in annotations {
            let ty = type_at_range(&db, FileRange { file_id, range });
            let actual = if display_source {
                let module = db.module_for_file(file_id);
                ty.display_source_code(&db, module).unwrap()
            } else {
                ty.display_test(&db).to_string()
            };
            assert_eq!(expected, actual);
            checked_one = true;
        }
    }
    assert!(checked_one, "no `//^` annotations found");
}

fn type_at_range(db: &TestDB, pos: FileRange) -> Ty {
    let file = db.parse(pos.file_id).ok().unwrap();
    let expr = algo::find_node_at_range::<ast::Expr>(file.syntax(), pos.range).unwrap();
    let fn_def = expr.syntax().ancestors().find_map(ast::Fn::cast).unwrap();
    let module = db.module_for_file(pos.file_id);
    let func = *module.child_by_source(db)[keys::FUNCTION]
        .get(&InFile::new(pos.file_id.into(), fn_def))
        .unwrap();

    let (_body, source_map) = db.body_with_source_map(func.into());
    if let Some(expr_id) = source_map.node_expr(InFile::new(pos.file_id.into(), &expr)) {
        let infer = db.infer(func.into());
        return infer[expr_id].clone();
    }
    panic!("Can't find expression")
}

fn infer(ra_fixture: &str) -> String {
    infer_with_mismatches(ra_fixture, false)
}

fn infer_with_mismatches(content: &str, include_mismatches: bool) -> String {
    let _tracing = setup_tracing();
    let (db, file_id) = TestDB::with_single_file(content);

    let mut buf = String::new();

    let mut infer_def = |inference_result: Arc<InferenceResult>,
                         body_source_map: Arc<BodySourceMap>| {
        let mut types: Vec<(InFile<SyntaxNode>, &Ty)> = Vec::new();
        let mut mismatches: Vec<(InFile<SyntaxNode>, &TypeMismatch)> = Vec::new();

        for (pat, ty) in inference_result.type_of_pat.iter() {
            let syntax_ptr = match body_source_map.pat_syntax(pat) {
                Ok(sp) => {
                    let root = db.parse_or_expand(sp.file_id).unwrap();
                    sp.map(|ptr| {
                        ptr.either(
                            |it| it.to_node(&root).syntax().clone(),
                            |it| it.to_node(&root).syntax().clone(),
                        )
                    })
                }
                Err(SyntheticSyntax) => continue,
            };
            types.push((syntax_ptr, ty));
        }

        for (expr, ty) in inference_result.type_of_expr.iter() {
            let node = match body_source_map.expr_syntax(expr) {
                Ok(sp) => {
                    let root = db.parse_or_expand(sp.file_id).unwrap();
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
                (node.value.text_range(), node.value.text().to_string().replace("\n", " "))
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
            let tree = db.item_tree(loc.id.file_id);
            tree.source(&db, loc.id).syntax().text_range().start()
        }
        DefWithBodyId::ConstId(it) => {
            let loc = it.lookup(&db);
            let tree = db.item_tree(loc.id.file_id);
            tree.source(&db, loc.id).syntax().text_range().start()
        }
        DefWithBodyId::StaticId(it) => {
            let loc = it.lookup(&db);
            let tree = db.item_tree(loc.id.file_id);
            tree.source(&db, loc.id).syntax().text_range().start()
        }
    });
    for def in defs {
        let (_body, source_map) = db.body_with_source_map(def);
        let infer = db.infer(def);
        infer_def(infer, source_map);
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
                    visit_scope(db, crate_def_map, &body.item_scope, cb);
                }
                AssocItemId::ConstId(it) => {
                    let def = it.into();
                    cb(def);
                    let body = db.body(def);
                    visit_scope(db, crate_def_map, &body.item_scope, cb);
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
                    visit_scope(db, crate_def_map, &body.item_scope, cb);
                }
                ModuleDefId::ConstId(it) => {
                    let def = it.into();
                    cb(def);
                    let body = db.body(def);
                    visit_scope(db, crate_def_map, &body.item_scope, cb);
                }
                ModuleDefId::StaticId(it) => {
                    let def = it.into();
                    cb(def);
                    let body = db.body(def);
                    visit_scope(db, crate_def_map, &body.item_scope, cb);
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

#[test]
fn typing_whitespace_inside_a_function_should_not_invalidate_types() {
    let (mut db, pos) = TestDB::with_position(
        "
        //- /lib.rs
        fn foo() -> i32 {
            $01 + 1
        }
    ",
    );
    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id);
            let crate_def_map = module.def_map(&db);
            visit_module(&db, &crate_def_map, module.local_id, &mut |def| {
                db.infer(def);
            });
        });
        assert!(format!("{:?}", events).contains("infer"))
    }

    let new_text = "
        fn foo() -> i32 {
            1
            +
            1
        }
    "
    .to_string();

    db.set_file_text(pos.file_id, Arc::new(new_text));

    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id);
            let crate_def_map = module.def_map(&db);
            visit_module(&db, &crate_def_map, module.local_id, &mut |def| {
                db.infer(def);
            });
        });
        assert!(!format!("{:?}", events).contains("infer"), "{:#?}", events)
    }
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
