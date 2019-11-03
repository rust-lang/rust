use std::sync::Arc;

use ra_db::{SourceDatabase, SourceDatabaseExt};

use super::*;

fn check_def_map_is_not_recomputed(initial: &str, file_change: &str) {
    let (mut db, pos) = TestDB::with_position(initial);
    let krate = db.crate_graph().iter().next().unwrap();
    {
        let events = db.log_executed(|| {
            db.crate_def_map(krate);
        });
        assert!(format!("{:?}", events).contains("crate_def_map"), "{:#?}", events)
    }
    db.set_file_text(pos.file_id, Arc::new(file_change.to_string()));

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
        "
        //- /lib.rs
        mod foo;<|>

        use crate::foo::bar::Baz;

        fn foo() -> i32 {
            1 + 1
        }
        //- /foo/mod.rs
        pub mod bar;

        //- /foo/bar.rs
        pub struct Baz;
        ",
        "
        mod foo;

        use crate::foo::bar::Baz;

        fn foo() -> i32 { 92 }
        ",
    );
}

#[test]
fn adding_inner_items_should_not_invalidate_def_map() {
    check_def_map_is_not_recomputed(
        "
        //- /lib.rs
        struct S { a: i32}
        enum E { A }
        trait T {
            fn a() {}
        }
        mod foo;<|>
        impl S {
            fn a() {}
        }
        use crate::foo::bar::Baz;
        //- /foo/mod.rs
        pub mod bar;

        //- /foo/bar.rs
        pub struct Baz;
        ",
        "
        struct S { a: i32, b: () }
        enum E { A, B }
        trait T {
            fn a() {}
            fn b() {}
        }
        mod foo;<|>
        impl S {
            fn a() {}
            fn b() {}
        }
        use crate::foo::bar::Baz;
        ",
    );
}

#[test]
fn typing_inside_a_macro_should_not_invalidate_def_map() {
    let (mut db, pos) = TestDB::with_position(
        "
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
    let krate = db.crate_graph().iter().next().unwrap();
    {
        let events = db.log_executed(|| {
            let crate_def_map = db.crate_def_map(krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.items.len(), 1);
        });
        assert!(format!("{:?}", events).contains("crate_def_map"), "{:#?}", events)
    }
    db.set_file_text(pos.file_id, Arc::new("m!(Y);".to_string()));

    {
        let events = db.log_executed(|| {
            let crate_def_map = db.crate_def_map(krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.items.len(), 1);
        });
        assert!(!format!("{:?}", events).contains("crate_def_map"), "{:#?}", events)
    }
}
