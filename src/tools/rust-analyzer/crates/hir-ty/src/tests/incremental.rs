use base_db::{fixture::WithFixture, SourceDatabaseExt};
use triomphe::Arc;

use crate::{db::HirDatabase, test_db::TestDB};

use super::visit_module;

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
        assert!(format!("{events:?}").contains("infer"))
    }

    let new_text = "
        fn foo() -> i32 {
            1
            +
            1
        }
    ";

    db.set_file_text(pos.file_id, Arc::from(new_text));

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
