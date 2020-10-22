use base_db::fixture::WithFixture;
use base_db::FileId;
use base_db::SourceDatabaseExt;
use hir_expand::db::AstDatabase;
use rustc_hash::FxHashMap;
use syntax::TextRange;
use syntax::TextSize;

use crate::test_db::TestDB;

fn check_diagnostics(ra_fixture: &str) {
    let db: TestDB = TestDB::with_files(ra_fixture);
    let annotations = db.extract_annotations();
    assert!(!annotations.is_empty());

    let mut actual: FxHashMap<FileId, Vec<(TextRange, String)>> = FxHashMap::default();
    db.diagnostics(|d| {
        let src = d.display_source();
        let root = db.parse_or_expand(src.file_id).unwrap();
        // FIXME: macros...
        let file_id = src.file_id.original_file(&db);
        let range = src.value.to_node(&root).text_range();
        let message = d.message().to_owned();
        actual.entry(file_id).or_default().push((range, message));
    });

    for (file_id, diags) in actual.iter_mut() {
        diags.sort_by_key(|it| it.0.start());
        let text = db.file_text(*file_id);
        // For multiline spans, place them on line start
        for (range, content) in diags {
            if text[*range].contains('\n') {
                *range = TextRange::new(range.start(), range.start() + TextSize::from(1));
                *content = format!("... {}", content);
            }
        }
    }

    assert_eq!(annotations, actual);
}

#[test]
fn unresolved_import() {
    check_diagnostics(
        r"
        use does_exist;
        use does_not_exist;
          //^^^^^^^^^^^^^^ unresolved import

        mod does_exist {}
        ",
    );
}

#[test]
fn unresolved_import_in_use_tree() {
    // Only the relevant part of a nested `use` item should be highlighted.
    check_diagnostics(
        r"
        use does_exist::{Exists, DoesntExist};
                               //^^^^^^^^^^^ unresolved import

        use {does_not_exist::*, does_exist};
           //^^^^^^^^^^^^^^^^^ unresolved import

        use does_not_exist::{
            a,
          //^ unresolved import
            b,
          //^ unresolved import
            c,
          //^ unresolved import
        };

        mod does_exist {
            pub struct Exists;
        }
        ",
    );
}

#[test]
fn unresolved_extern_crate() {
    check_diagnostics(
        r"
        //- /main.rs crate:main deps:core
        extern crate core;
          extern crate doesnotexist;
        //^^^^^^^^^^^^^^^^^^^^^^^^^^ unresolved extern crate
        //- /lib.rs crate:core
        ",
    );
}

#[test]
fn dedup_unresolved_import_from_unresolved_crate() {
    check_diagnostics(
        r"
        //- /main.rs crate:main
        mod a {
            extern crate doesnotexist;
          //^^^^^^^^^^^^^^^^^^^^^^^^^^ unresolved extern crate

            // Should not error, since we already errored for the missing crate.
            use doesnotexist::{self, bla, *};

            use crate::doesnotexist;
              //^^^^^^^^^^^^^^^^^^^ unresolved import
        }

        mod m {
            use super::doesnotexist;
              //^^^^^^^^^^^^^^^^^^^ unresolved import
        }
        ",
    );
}

#[test]
fn unresolved_module() {
    check_diagnostics(
        r"
        //- /lib.rs
        mod foo;
          mod bar;
        //^^^^^^^^ unresolved module
        mod baz {}
        //- /foo.rs
        ",
    );
}

#[test]
fn inactive_item() {
    // Additional tests in `cfg` crate. This only tests disabled cfgs.

    check_diagnostics(
        r#"
        //- /lib.rs
          #[cfg(no)] pub fn f() {}
        //^^^^^^^^^^^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: no is disabled

          #[cfg(no)] #[cfg(no2)] mod m;
        //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: no and no2 are disabled

          #[cfg(all(not(a), b))] enum E {}
        //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: b is disabled

          #[cfg(feature = "std")] use std;
        //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ code is inactive due to #[cfg] directives: feature = "std" is disabled
        "#,
    );
}
