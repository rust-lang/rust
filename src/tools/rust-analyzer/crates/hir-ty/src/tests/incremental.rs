use base_db::SourceDatabase;
use expect_test::Expect;
use hir_def::{DefWithBodyId, ModuleDefId};
use salsa::EventKind;
use test_fixture::WithFixture;

use crate::{db::HirDatabase, test_db::TestDB};

use super::visit_module;

#[test]
fn typing_whitespace_inside_a_function_should_not_invalidate_types() {
    let (mut db, pos) = TestDB::with_position(
        "
//- /lib.rs
fn foo() -> i32 {
    $01 + 1
}",
    );
    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let crate_def_map = module.def_map(&db);
            visit_module(&db, crate_def_map, module.local_id, &mut |def| {
                if let ModuleDefId::FunctionId(it) = def {
                    db.infer(it.into());
                }
            });
        },
        &[("infer_shim", 1)],
        expect_test::expect![[r#"
            [
                "source_root_crates_shim",
                "crate_local_def_map",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "infer_shim",
                "function_signature_shim",
                "function_signature_with_source_map_shim",
                "attrs_shim",
                "body_shim",
                "body_with_source_map_shim",
                "trait_environment_shim",
                "return_type_impl_traits_shim",
                "expr_scopes_shim",
                "lang_item",
                "crate_lang_items",
                "lang_item",
            ]
        "#]],
    );

    let new_text = "
fn foo() -> i32 {
    1
    +
    1
}";

    db.set_file_text(pos.file_id.file_id(&db), new_text);

    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let crate_def_map = module.def_map(&db);
            visit_module(&db, crate_def_map, module.local_id, &mut |def| {
                if let ModuleDefId::FunctionId(it) = def {
                    db.infer(it.into());
                }
            });
        },
        &[("infer_shim", 0)],
        expect_test::expect![[r#"
            [
                "parse_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
                "real_span_map_shim",
                "attrs_shim",
                "function_signature_with_source_map_shim",
                "function_signature_shim",
                "body_with_source_map_shim",
                "body_shim",
            ]
        "#]],
    );
}

#[test]
fn typing_inside_a_function_should_not_invalidate_types_in_another() {
    let (mut db, pos) = TestDB::with_position(
        "
//- /lib.rs
fn foo() -> f32 {
    1.0 + 2.0
}
fn bar() -> i32 {
    $01 + 1
}
fn baz() -> i32 {
    1 + 1
}",
    );
    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let crate_def_map = module.def_map(&db);
            visit_module(&db, crate_def_map, module.local_id, &mut |def| {
                if let ModuleDefId::FunctionId(it) = def {
                    db.infer(it.into());
                }
            });
        },
        &[("infer_shim", 3)],
        expect_test::expect![[r#"
            [
                "source_root_crates_shim",
                "crate_local_def_map",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "infer_shim",
                "function_signature_shim",
                "function_signature_with_source_map_shim",
                "attrs_shim",
                "body_shim",
                "body_with_source_map_shim",
                "trait_environment_shim",
                "return_type_impl_traits_shim",
                "expr_scopes_shim",
                "lang_item",
                "crate_lang_items",
                "attrs_shim",
                "attrs_shim",
                "lang_item",
                "infer_shim",
                "function_signature_shim",
                "function_signature_with_source_map_shim",
                "body_shim",
                "body_with_source_map_shim",
                "trait_environment_shim",
                "return_type_impl_traits_shim",
                "expr_scopes_shim",
                "infer_shim",
                "function_signature_shim",
                "function_signature_with_source_map_shim",
                "body_shim",
                "body_with_source_map_shim",
                "trait_environment_shim",
                "return_type_impl_traits_shim",
                "expr_scopes_shim",
            ]
        "#]],
    );

    let new_text = "
fn foo() -> f32 {
    1.0 + 2.0
}
fn bar() -> i32 {
    53
}
fn baz() -> i32 {
    1 + 1
}
";

    db.set_file_text(pos.file_id.file_id(&db), new_text);

    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let crate_def_map = module.def_map(&db);
            visit_module(&db, crate_def_map, module.local_id, &mut |def| {
                if let ModuleDefId::FunctionId(it) = def {
                    db.infer(it.into());
                }
            });
        },
        &[("infer_shim", 1)],
        expect_test::expect![[r#"
            [
                "parse_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
                "real_span_map_shim",
                "attrs_shim",
                "function_signature_with_source_map_shim",
                "function_signature_shim",
                "body_with_source_map_shim",
                "body_shim",
                "attrs_shim",
                "attrs_shim",
                "function_signature_with_source_map_shim",
                "function_signature_shim",
                "body_with_source_map_shim",
                "body_shim",
                "infer_shim",
                "expr_scopes_shim",
                "function_signature_with_source_map_shim",
                "function_signature_shim",
                "body_with_source_map_shim",
                "body_shim",
            ]
        "#]],
    );
}

#[test]
fn adding_struct_invalidates_infer() {
    let (mut db, pos) = TestDB::with_position(
        "
//- /lib.rs
fn foo() -> i32 {
    1 + 1
}

fn bar() -> f32 {
    2.0 * 3.0
}
$0",
    );
    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        },
        &[("trait_impls_in_crate_shim", 1)],
        expect_test::expect![[r#"
            [
                "source_root_crates_shim",
                "crate_local_def_map",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "trait_impls_in_crate_shim",
            ]
        "#]],
    );

    let new_text = "
fn foo() -> i32 {
    1 + 1
}

fn bar() -> f32 {
    2.0 * 3.0
}

pub struct NewStruct {
    field: i32,
}
";

    db.set_file_text(pos.file_id.file_id(&db), new_text);

    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        },
        &[("trait_impls_in_crate_shim", 1)],
        expect_test::expect![[r#"
            [
                "parse_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
                "real_span_map_shim",
                "crate_local_def_map",
                "trait_impls_in_crate_shim",
            ]
        "#]],
    );
}

#[test]
fn adding_enum_query_log() {
    let (mut db, pos) = TestDB::with_position(
        "
//- /lib.rs
fn foo() -> i32 {
    1 + 1
}

fn bar() -> f32 {
    2.0 * 3.0
}
$0",
    );
    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        },
        &[("trait_impls_in_crate_shim", 1)],
        expect_test::expect![[r#"
            [
                "source_root_crates_shim",
                "crate_local_def_map",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "trait_impls_in_crate_shim",
            ]
        "#]],
    );

    let new_text = "
fn foo() -> i32 {
    1 + 1
}

fn bar() -> f32 {
    2.0 * 3.0
}

pub enum SomeEnum {
    A,
    B
}
";

    db.set_file_text(pos.file_id.file_id(&db), new_text);

    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        },
        &[("trait_impls_in_crate_shim", 1)],
        expect_test::expect![[r#"
            [
                "parse_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
                "real_span_map_shim",
                "crate_local_def_map",
                "trait_impls_in_crate_shim",
            ]
        "#]],
    );
}

#[test]
fn adding_use_query_log() {
    let (mut db, pos) = TestDB::with_position(
        "
//- /lib.rs
fn foo() -> i32 {
    1 + 1
}

fn bar() -> f32 {
    2.0 * 3.0
}
$0",
    );
    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        },
        &[("trait_impls_in_crate_shim", 1)],
        expect_test::expect![[r#"
            [
                "source_root_crates_shim",
                "crate_local_def_map",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "trait_impls_in_crate_shim",
            ]
        "#]],
    );

    let new_text = "
use std::collections::HashMap;

fn foo() -> i32 {
    1 + 1
}

fn bar() -> f32 {
    2.0 * 3.0
}
";

    db.set_file_text(pos.file_id.file_id(&db), new_text);

    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        },
        &[("trait_impls_in_crate_shim", 1)],
        expect_test::expect![[r#"
            [
                "parse_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
                "real_span_map_shim",
                "crate_local_def_map",
                "trait_impls_in_crate_shim",
            ]
        "#]],
    );
}

#[test]
fn adding_impl_query_log() {
    let (mut db, pos) = TestDB::with_position(
        "
//- /lib.rs
fn foo() -> i32 {
    1 + 1
}

fn bar() -> f32 {
    2.0 * 3.0
}

pub struct SomeStruct {
    field: i32,
}
$0",
    );
    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        },
        &[("trait_impls_in_crate_shim", 1)],
        expect_test::expect![[r#"
            [
                "source_root_crates_shim",
                "crate_local_def_map",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "trait_impls_in_crate_shim",
            ]
        "#]],
    );

    let new_text = "
fn foo() -> i32 {
    1 + 1
}

fn bar() -> f32 {
    2.0 * 3.0
}

pub struct SomeStruct {
    field: i32,
}

impl SomeStruct {
    pub fn new(value: i32) -> Self {
        Self { field: value }
    }
}
";

    db.set_file_text(pos.file_id.file_id(&db), new_text);

    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        },
        &[("trait_impls_in_crate_shim", 1)],
        expect_test::expect![[r#"
            [
                "parse_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
                "real_span_map_shim",
                "crate_local_def_map",
                "trait_impls_in_crate_shim",
                "attrs_shim",
                "impl_trait_with_diagnostics_shim",
                "impl_signature_shim",
                "impl_signature_with_source_map_shim",
                "impl_self_ty_with_diagnostics_shim",
                "struct_signature_shim",
                "struct_signature_with_source_map_shim",
                "attrs_shim",
                "type_for_adt_tracked",
            ]
        "#]],
    );
}

#[test]
fn add_struct_invalidates_trait_solve() {
    let (mut db, file_id) = TestDB::with_single_file(
        "
//- /main.rs crate:main
struct SomeStruct;

trait Trait<T> {
    fn method(&self) -> T;
}
impl Trait<u32> for SomeStruct {}

fn main() {
    let s = SomeStruct;
    s.method();
    s.$0
}",
    );

    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(file_id.file_id(&db));
            let crate_def_map = module.def_map(&db);
            let mut defs: Vec<DefWithBodyId> = vec![];
            visit_module(&db, crate_def_map, module.local_id, &mut |it| {
                let def = match it {
                    ModuleDefId::FunctionId(it) => it.into(),
                    ModuleDefId::EnumVariantId(it) => it.into(),
                    ModuleDefId::ConstId(it) => it.into(),
                    ModuleDefId::StaticId(it) => it.into(),
                    _ => return,
                };
                defs.push(def);
            });

            for def in defs {
                let _inference_result = db.infer(def);
            }
        },
        &[("trait_solve_shim", 2)],
        expect_test::expect![[r#"
            [
                "source_root_crates_shim",
                "crate_local_def_map",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "TraitItems::query_with_diagnostics_",
                "body_shim",
                "body_with_source_map_shim",
                "attrs_shim",
                "ImplItems::of_",
                "infer_shim",
                "trait_signature_shim",
                "trait_signature_with_source_map_shim",
                "attrs_shim",
                "function_signature_shim",
                "function_signature_with_source_map_shim",
                "attrs_shim",
                "body_shim",
                "body_with_source_map_shim",
                "trait_environment_shim",
                "lang_item",
                "crate_lang_items",
                "attrs_shim",
                "attrs_shim",
                "return_type_impl_traits_shim",
                "infer_shim",
                "function_signature_shim",
                "function_signature_with_source_map_shim",
                "trait_environment_shim",
                "expr_scopes_shim",
                "struct_signature_shim",
                "struct_signature_with_source_map_shim",
                "generic_predicates_shim",
                "value_ty_shim",
                "VariantFields::firewall_",
                "VariantFields::query_",
                "lang_item",
                "inherent_impls_in_crate_shim",
                "impl_signature_shim",
                "impl_signature_with_source_map_shim",
                "callable_item_signature_shim",
                "adt_variance_shim",
                "variances_of_shim",
                "trait_solve_shim",
                "trait_datum_shim",
                "generic_predicates_shim",
                "adt_datum_shim",
                "trait_impls_in_deps_shim",
                "trait_impls_in_crate_shim",
                "impl_trait_with_diagnostics_shim",
                "impl_self_ty_with_diagnostics_shim",
                "type_for_adt_tracked",
                "impl_datum_shim",
                "generic_predicates_shim",
                "program_clauses_for_chalk_env_shim",
                "value_ty_shim",
                "generic_predicates_shim",
                "trait_solve_shim",
                "lang_item",
            ]
        "#]],
    );

    let new_text = "
//- /main.rs crate:main
struct AnotherStruct;

struct SomeStruct;

trait Trait<T> {
    fn method(&self) -> T;
}
impl Trait<u32> for SomeStruct {}

fn main() {
    let s = SomeStruct;
    s.method();
    s.$0
}";

    db.set_file_text(file_id.file_id(&db), new_text);

    execute_assert_events(
        &db,
        || {
            let module = db.module_for_file(file_id.file_id(&db));
            let crate_def_map = module.def_map(&db);
            let mut defs: Vec<DefWithBodyId> = vec![];

            visit_module(&db, crate_def_map, module.local_id, &mut |it| {
                let def = match it {
                    ModuleDefId::FunctionId(it) => it.into(),
                    ModuleDefId::EnumVariantId(it) => it.into(),
                    ModuleDefId::ConstId(it) => it.into(),
                    ModuleDefId::StaticId(it) => it.into(),
                    _ => return,
                };
                defs.push(def);
            });

            for def in defs {
                let _inference_result = db.infer(def);
            }
        },
        &[("trait_solve_shim", 0)],
        expect_test::expect![[r#"
            [
                "parse_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
                "real_span_map_shim",
                "crate_local_def_map",
                "TraitItems::query_with_diagnostics_",
                "body_with_source_map_shim",
                "attrs_shim",
                "body_shim",
                "ImplItems::of_",
                "infer_shim",
                "attrs_shim",
                "trait_signature_with_source_map_shim",
                "attrs_shim",
                "function_signature_with_source_map_shim",
                "function_signature_shim",
                "body_with_source_map_shim",
                "body_shim",
                "trait_environment_shim",
                "crate_lang_items",
                "attrs_shim",
                "attrs_shim",
                "attrs_shim",
                "return_type_impl_traits_shim",
                "infer_shim",
                "function_signature_with_source_map_shim",
                "expr_scopes_shim",
                "struct_signature_with_source_map_shim",
                "VariantFields::query_",
                "inherent_impls_in_crate_shim",
                "impl_signature_with_source_map_shim",
                "impl_signature_shim",
                "callable_item_signature_shim",
                "generic_predicates_shim",
                "trait_impls_in_crate_shim",
                "impl_trait_with_diagnostics_shim",
                "impl_self_ty_with_diagnostics_shim",
                "generic_predicates_shim",
            ]
        "#]],
    );
}

fn execute_assert_events(
    db: &TestDB,
    f: impl FnOnce(),
    required: &[(&str, usize)],
    expect: Expect,
) {
    let (executed, events) = db.log_executed(f);
    salsa::attach(db, || {
        for (event, count) in required {
            let n = executed.iter().filter(|it| it.contains(event)).count();
            assert_eq!(
                n,
                *count,
                "Expected {event} to be executed {count} times, but only got {n}:\n \
             Executed: {executed:#?}\n \
             Event log: {events:#?}",
                events = events
                    .iter()
                    .filter(|event| !matches!(event.kind, EventKind::WillCheckCancellation))
                    .map(|event| { format!("{:?}", event.kind) })
                    .collect::<Vec<_>>(),
            );
        }
        expect.assert_debug_eq(&executed);
    });
}
