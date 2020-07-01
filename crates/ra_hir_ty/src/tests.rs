mod never_type;
mod coercion;
mod regression;
mod simple;
mod patterns;
mod traits;
mod method_resolution;
mod macros;
mod display_source_code;

use std::sync::Arc;

use hir_def::{
    body::{BodySourceMap, SyntheticSyntax},
    child_by_source::ChildBySource,
    db::DefDatabase,
    item_scope::ItemScope,
    keys,
    nameres::CrateDefMap,
    AssocItemId, DefWithBodyId, LocalModuleId, Lookup, ModuleDefId,
};
use hir_expand::{db::AstDatabase, InFile};
use insta::assert_snapshot;
use ra_db::{fixture::WithFixture, salsa::Database, FileRange, SourceDatabase};
use ra_syntax::{
    algo,
    ast::{self, AstNode},
    SyntaxNode,
};
use stdx::format_to;

use crate::{
    db::HirDatabase, display::HirDisplay, infer::TypeMismatch, test_db::TestDB, InferenceResult, Ty,
};

// These tests compare the inference results for all expressions in a file
// against snapshots of the expected results using insta. Use cargo-insta to
// update the snapshots.

fn check_types(ra_fixture: &str) {
    check_types_impl(ra_fixture, false)
}

fn check_types_source_code(ra_fixture: &str) {
    check_types_impl(ra_fixture, true)
}

fn check_types_impl(ra_fixture: &str, display_source: bool) {
    let db = TestDB::with_files(ra_fixture);
    let mut checked_one = false;
    for (file_id, annotations) in db.extract_annotations() {
        for (range, expected) in annotations {
            let ty = type_at_range(&db, FileRange { file_id, range });
            let actual = if display_source {
                let module = db.module_for_file(file_id);
                ty.display_source_code(&db, module).unwrap()
            } else {
                ty.display(&db).to_string()
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
    let fn_def = expr.syntax().ancestors().find_map(ast::FnDef::cast).unwrap();
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
                (self_param.self_token().unwrap().text_range(), "self".to_string())
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
                ty.display(&db)
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
                    mismatch.expected.display(&db),
                    mismatch.actual.display(&db),
                );
            }
        }
    };

    let module = db.module_for_file(file_id);
    let crate_def_map = db.crate_def_map(module.krate);

    let mut defs: Vec<DefWithBodyId> = Vec::new();
    visit_module(&db, &crate_def_map, module.local_id, &mut |it| defs.push(it));
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
    crate_def_map: &CrateDefMap,
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
        crate_def_map: &CrateDefMap,
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
            <|>1 + 1
        }
    ",
    );
    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id);
            let crate_def_map = db.crate_def_map(module.krate);
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

    db.query_mut(ra_db::FileTextQuery).set(pos.file_id, Arc::new(new_text));

    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id);
            let crate_def_map = db.crate_def_map(module.krate);
            visit_module(&db, &crate_def_map, module.local_id, &mut |def| {
                db.infer(def);
            });
        });
        assert!(!format!("{:?}", events).contains("infer"), "{:#?}", events)
    }
}

#[test]
fn no_such_field_diagnostics() {
    let diagnostics = TestDB::with_files(
        r"
        //- /lib.rs
        struct S { foo: i32, bar: () }
        impl S {
            fn new() -> S {
                S {
                    foo: 92,
                    baz: 62,
                }
            }
        }
        ",
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @r###"
    "baz: 62": no such field
    "{\n            foo: 92,\n            baz: 62,\n        }": Missing structure fields:
    - bar
    "###
    );
}

#[test]
fn no_such_field_with_feature_flag_diagnostics() {
    let diagnostics = TestDB::with_files(
        r#"
        //- /lib.rs crate:foo cfg:feature=foo
        struct MyStruct {
            my_val: usize,
            #[cfg(feature = "foo")]
            bar: bool,
        }

        impl MyStruct {
            #[cfg(feature = "foo")]
            pub(crate) fn new(my_val: usize, bar: bool) -> Self {
                Self { my_val, bar }
            }

            #[cfg(not(feature = "foo"))]
            pub(crate) fn new(my_val: usize, _bar: bool) -> Self {
                Self { my_val }
            }
        }
        "#,
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @r###""###);
}

#[test]
fn no_such_field_enum_with_feature_flag_diagnostics() {
    let diagnostics = TestDB::with_files(
        r#"
        //- /lib.rs crate:foo cfg:feature=foo
        enum Foo {
            #[cfg(not(feature = "foo"))]
            Buz,
            #[cfg(feature = "foo")]
            Bar,
            Baz
        }

        fn test_fn(f: Foo) {
            match f {
                Foo::Bar => {},
                Foo::Baz => {},
            }
        }
        "#,
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @r###""###);
}

#[test]
fn no_such_field_with_feature_flag_diagnostics_on_struct_lit() {
    let diagnostics = TestDB::with_files(
        r#"
        //- /lib.rs crate:foo cfg:feature=foo
        struct S {
            #[cfg(feature = "foo")]
            foo: u32,
            #[cfg(not(feature = "foo"))]
            bar: u32,
        }

        impl S {
            #[cfg(feature = "foo")]
            fn new(foo: u32) -> Self {
                Self { foo }
            }
            #[cfg(not(feature = "foo"))]
            fn new(bar: u32) -> Self {
                Self { bar }
            }
        }
        "#,
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @r###""###);
}

#[test]
fn no_such_field_with_feature_flag_diagnostics_on_block_expr() {
    let diagnostics = TestDB::with_files(
        r#"
        //- /lib.rs crate:foo cfg:feature=foo
        struct S {
            #[cfg(feature = "foo")]
            foo: u32,
            #[cfg(not(feature = "foo"))]
            bar: u32,
        }

        impl S {
            fn new(bar: u32) -> Self {
                #[cfg(feature = "foo")]
                {
                Self { foo: bar }
                }
                #[cfg(not(feature = "foo"))]
                {
                Self { bar }
                }
            }
        }
        "#,
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @r###""###);
}

#[test]
fn no_such_field_with_feature_flag_diagnostics_on_struct_fields() {
    let diagnostics = TestDB::with_files(
        r#"
        //- /lib.rs crate:foo cfg:feature=foo
        struct S {
            #[cfg(feature = "foo")]
            foo: u32,
            #[cfg(not(feature = "foo"))]
            bar: u32,
        }

        impl S {
            fn new(val: u32) -> Self {
                Self {
                    #[cfg(feature = "foo")]
                    foo: val,
                    #[cfg(not(feature = "foo"))]
                    bar: val,
                }
            }
        }
        "#,
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @r###""###);
}

#[test]
fn no_such_field_with_type_macro() {
    let diagnostics = TestDB::with_files(
        r"
        macro_rules! Type {
            () => { u32 };
        }

        struct Foo {
            bar: Type![],
        }
        impl Foo {
            fn new() -> Self {
                Foo { bar: 0 }
            }
        }
        ",
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @r###""###);
}

#[test]
fn missing_record_pat_field_diagnostic() {
    let diagnostics = TestDB::with_files(
        r"
        //- /lib.rs
        struct S { foo: i32, bar: () }
        fn baz(s: S) {
            let S { foo: _ } = s;
        }
        ",
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @r###"
    "{ foo: _ }": Missing structure fields:
    - bar
    "###
    );
}

#[test]
fn missing_record_pat_field_no_diagnostic_if_not_exhaustive() {
    let diagnostics = TestDB::with_files(
        r"
        //- /lib.rs
        struct S { foo: i32, bar: () }
        fn baz(s: S) -> i32 {
            match s {
                S { foo, .. } => foo,
            }
        }
        ",
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @"");
}

#[test]
fn missing_unsafe_diagnostic_with_raw_ptr() {
    let diagnostics = TestDB::with_files(
        r"
//- /lib.rs
fn missing_unsafe() {
    let x = &5 as *const usize;
    let y = *x;
}
",
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @r#""*x": This operation is unsafe and requires an unsafe function or block"#);
}

#[test]
fn missing_unsafe_diagnostic_with_unsafe_call() {
    let diagnostics = TestDB::with_files(
        r"
//- /lib.rs
unsafe fn unsafe_fn() {
    let x = &5 as *const usize;
    let y = *x;
}

fn missing_unsafe() {
    unsafe_fn();
}
",
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @r#""unsafe_fn()": This operation is unsafe and requires an unsafe function or block"#);
}

#[test]
fn missing_unsafe_diagnostic_with_unsafe_method_call() {
    let diagnostics = TestDB::with_files(
        r"
struct HasUnsafe;

impl HasUnsafe {
    unsafe fn unsafe_fn(&self) {
        let x = &5 as *const usize;
        let y = *x;
    }
}

fn missing_unsafe() {
    HasUnsafe.unsafe_fn();
}

",
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @r#""HasUnsafe.unsafe_fn()": This operation is unsafe and requires an unsafe function or block"#);
}

#[test]
fn no_missing_unsafe_diagnostic_with_raw_ptr_in_unsafe_block() {
    let diagnostics = TestDB::with_files(
        r"
fn nothing_to_see_move_along() {
    let x = &5 as *const usize;
    unsafe {
        let y = *x;
    }
}
",
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @"");
}

#[test]
fn missing_unsafe_diagnostic_with_raw_ptr_outside_unsafe_block() {
    let diagnostics = TestDB::with_files(
        r"
fn nothing_to_see_move_along() {
    let x = &5 as *const usize;
    unsafe {
        let y = *x;
    }
    let z = *x;
}
",
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @r#""*x": This operation is unsafe and requires an unsafe function or block"#);
}

#[test]
fn no_missing_unsafe_diagnostic_with_unsafe_call_in_unsafe_block() {
    let diagnostics = TestDB::with_files(
        r"
unsafe fn unsafe_fn() {
    let x = &5 as *const usize;
    let y = *x;
}

fn nothing_to_see_move_along() {
    unsafe {
        unsafe_fn();
    }
}
",
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @"");
}

#[test]
fn no_missing_unsafe_diagnostic_with_unsafe_method_call_in_unsafe_block() {
    let diagnostics = TestDB::with_files(
        r"
struct HasUnsafe;

impl HasUnsafe {
    unsafe fn unsafe_fn() {
        let x = &5 as *const usize;
        let y = *x;
    }
}

fn nothing_to_see_move_along() {
    unsafe {
        HasUnsafe.unsafe_fn();
    }
}

",
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @"");
}

#[test]
fn break_outside_of_loop() {
    let diagnostics = TestDB::with_files(
        r"
        //- /lib.rs
        fn foo() {
            break;
        }
        ",
    )
    .diagnostics()
    .0;

    assert_snapshot!(diagnostics, @r###""break": break outside of loop
    "###
    );
}
