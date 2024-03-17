use base_db::SourceDatabaseExt2 as _;
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
            let module = db.module_for_file(pos.file_id);
            let crate_def_map = module.def_map(&db);
            visit_module(&db, &crate_def_map, module.local_id, &mut |def| {
                db.infer(def);
            });
        });
        assert!(format!("{events:?}").contains("infer"))
    }

    let new_text = "
fn foo() -> i32 {
    1
    +
    1
}";

    db.set_file_text(pos.file_id, new_text);

    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id);
            let crate_def_map = module.def_map(&db);
            visit_module(&db, &crate_def_map, module.local_id, &mut |def| {
                db.infer(def);
            });
        });
        assert!(!format!("{events:?}").contains("infer"), "{events:#?}")
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
            let module = db.module_for_file(pos.file_id);
            let crate_def_map = module.def_map(&db);
            visit_module(&db, &crate_def_map, module.local_id, &mut |def| {
                db.infer(def);
            });
        });
        assert!(format!("{events:?}").contains("infer"))
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

    db.set_file_text(pos.file_id, new_text);

    {
        let events = db.log_executed(|| {
            let module = db.module_for_file(pos.file_id);
            let crate_def_map = module.def_map(&db);
            visit_module(&db, &crate_def_map, module.local_id, &mut |def| {
                db.infer(def);
            });
        });
        assert!(format!("{events:?}").matches("infer").count() == 1, "{events:#?}")
    }
}
