use std::sync::Arc;

use ra_db::SourceDatabaseExt;

use super::*;

fn check_def_map_is_not_recomputed(ra_fixture_initial: &str, ra_fixture_change: &str) {
    let (mut db, pos) = TestDB::with_position(ra_fixture_initial);
    let krate = db.test_crate();
    {
        let events = db.log_executed(|| {
            db.crate_def_map(krate);
        });
        assert!(format!("{:?}", events).contains("crate_def_map"), "{:#?}", events)
    }
    db.set_file_text(pos.file_id, Arc::new(ra_fixture_change.to_string()));

    {
        let events = db.log_executed(|| {
            db.crate_def_map(krate);
        });
        assert!(!format!("{:?}", events).contains("crate_def_map"), "{:#?}", events)
    }
}

#[test]
fn typing_inside_a_function_should_not_invalidate_def_map() {
    check_def_map_is_not_recomputed(
        r"
        //- /lib.rs
        mod foo;<|>

        use crate::foo::bar::Baz;

        enum E { A, B }
        use E::*;

        fn foo() -> i32 {
            1 + 1
        }
        //- /foo/mod.rs
        pub mod bar;

        //- /foo/bar.rs
        pub struct Baz;
        ",
        r"
        mod foo;

        use crate::foo::bar::Baz;

        enum E { A, B }
        use E::*;

        fn foo() -> i32 { 92 }
        ",
    );
}

#[test]
fn typing_inside_a_macro_should_not_invalidate_def_map() {
    let (mut db, pos) = TestDB::with_position(
        r"
        //- /lib.rs
        macro_rules! m {
            ($ident:ident) => {
                fn f() {
                    $ident + $ident;
                };
            }
        }
        mod foo;

        //- /foo/mod.rs
        pub mod bar;

        //- /foo/bar.rs
        <|>
        m!(X);
        ",
    );
    let krate = db.test_crate();
    {
        let events = db.log_executed(|| {
            let crate_def_map = db.crate_def_map(krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 1);
        });
        assert!(format!("{:?}", events).contains("crate_def_map"), "{:#?}", events)
    }
    db.set_file_text(pos.file_id, Arc::new("m!(Y);".to_string()));

    {
        let events = db.log_executed(|| {
            let crate_def_map = db.crate_def_map(krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 1);
        });
        assert!(!format!("{:?}", events).contains("crate_def_map"), "{:#?}", events)
    }
}
