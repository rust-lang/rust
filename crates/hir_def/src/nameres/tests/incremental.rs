use std::sync::Arc;

use base_db::SourceDatabaseExt;

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
        mod foo;$0

        use crate::foo::bar::Baz;

        enum E { A, B }
        use E::*;

        fn foo() -> i32 {
            1 + 1
        }

        #[cfg(never)]
        fn no() {}
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

        #[cfg(never)]
        fn no() {}
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
        $0
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

#[test]
fn typing_inside_a_function_should_not_invalidate_expansions() {
    let (mut db, pos) = TestDB::with_position(
        r#"
//- /lib.rs
macro_rules! m {
    ($ident:ident) => {
        fn $ident() { };
    }
}
mod foo;

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
m!(X);
fn quux() { 1$0 }
m!(Y);
m!(Z);
"#,
    );
    let krate = db.test_crate();
    {
        let events = db.log_executed(|| {
            let crate_def_map = db.crate_def_map(krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 4);
        });
        let n_recalculated_item_trees = events.iter().filter(|it| it.contains("item_tree")).count();
        assert_eq!(n_recalculated_item_trees, 6);
    }

    let new_text = r#"
m!(X);
fn quux() { 92 }
m!(Y);
m!(Z);
"#;
    db.set_file_text(pos.file_id, Arc::new(new_text.to_string()));

    {
        let events = db.log_executed(|| {
            let crate_def_map = db.crate_def_map(krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 4);
        });
        let n_recalculated_item_trees = events.iter().filter(|it| it.contains("item_tree")).count();
        assert_eq!(n_recalculated_item_trees, 1);
    }
}
