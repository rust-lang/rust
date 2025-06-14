use base_db::SourceDatabase;
use hir_def::ModuleDefId;
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
    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let crate_def_map = module.def_map(&db);
            visit_module(&db, crate_def_map, module.local_id, &mut |def| {
                if let ModuleDefId::FunctionId(it) = def {
                    db.infer(it.into());
                }
            });
        });
        assert!(format!("{events:?}").contains("infer_shim"))
    }

    let new_text = "
fn foo() -> i32 {
    1
    +
    1
}";

    db.set_file_text(pos.file_id.file_id(&db), new_text);

    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let crate_def_map = module.def_map(&db);
            visit_module(&db, crate_def_map, module.local_id, &mut |def| {
                if let ModuleDefId::FunctionId(it) = def {
                    db.infer(it.into());
                }
            });
        });
        assert!(!format!("{events:?}").contains("infer_shim"), "{events:#?}")
    }
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
    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let crate_def_map = module.def_map(&db);
            visit_module(&db, crate_def_map, module.local_id, &mut |def| {
                if let ModuleDefId::FunctionId(it) = def {
                    db.infer(it.into());
                }
            });
        });
        assert!(format!("{events:?}").contains("infer_shim"))
    }

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

    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let crate_def_map = module.def_map(&db);
            visit_module(&db, crate_def_map, module.local_id, &mut |def| {
                if let ModuleDefId::FunctionId(it) = def {
                    db.infer(it.into());
                }
            });
        });
        assert_eq!(format!("{events:?}").matches("infer_shim").count(), 1, "{events:#?}")
    }
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
    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        });
        assert!(format!("{events:?}").contains("trait_impls_in_crate_shim"))
    }

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

    {
        let actual = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        });

        let expected = vec![
            "parse_shim".to_owned(),
            "ast_id_map_shim".to_owned(),
            "file_item_tree_shim".to_owned(),
            "real_span_map_shim".to_owned(),
            "crate_local_def_map".to_owned(),
            "trait_impls_in_crate_shim".to_owned(),
        ];

        assert_eq!(expected, actual);
    }
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
    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        });
        assert!(format!("{events:?}").contains("trait_impls_in_crate_shim"))
    }

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

    {
        let actual = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        });

        let expected = vec![
            "parse_shim".to_owned(),
            "ast_id_map_shim".to_owned(),
            "file_item_tree_shim".to_owned(),
            "real_span_map_shim".to_owned(),
            "crate_local_def_map".to_owned(),
            "trait_impls_in_crate_shim".to_owned(),
        ];

        assert_eq!(expected, actual);
    }
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
    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        });
        assert!(format!("{events:?}").contains("trait_impls_in_crate_shim"))
    }

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

    {
        let actual = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        });

        let expected = vec![
            "parse_shim".to_owned(),
            "ast_id_map_shim".to_owned(),
            "file_item_tree_shim".to_owned(),
            "real_span_map_shim".to_owned(),
            "crate_local_def_map".to_owned(),
            "trait_impls_in_crate_shim".to_owned(),
        ];

        assert_eq!(expected, actual);
    }
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
    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        });
        assert!(format!("{events:?}").contains("trait_impls_in_crate_shim"))
    }

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

    {
        let actual = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id.file_id(&db));
            let _crate_def_map = module.def_map(&db);
            db.trait_impls_in_crate(module.krate());
        });

        let expected = vec![
            "parse_shim".to_owned(),
            "ast_id_map_shim".to_owned(),
            "file_item_tree_shim".to_owned(),
            "real_span_map_shim".to_owned(),
            "crate_local_def_map".to_owned(),
            "trait_impls_in_crate_shim".to_owned(),
            "attrs_shim".to_owned(),
            "impl_trait_with_diagnostics_shim".to_owned(),
            "impl_signature_shim".to_owned(),
            "impl_signature_with_source_map_shim".to_owned(),
            "impl_self_ty_with_diagnostics_shim".to_owned(),
            "struct_signature_shim".to_owned(),
            "struct_signature_with_source_map_shim".to_owned(),
            "type_for_adt_tracked".to_owned(),
        ];

        assert_eq!(expected, actual);
    }
}
