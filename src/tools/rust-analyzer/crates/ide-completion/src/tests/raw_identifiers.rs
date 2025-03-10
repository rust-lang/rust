use base_db::SourceDatabase;
use expect_test::{Expect, expect};
use itertools::Itertools;

use crate::tests::{TEST_CONFIG, completion_list_with_config_raw, position};

fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let completions = completion_list_with_config_raw(TEST_CONFIG, ra_fixture, true, None);
    let (db, position) = position(ra_fixture);
    let mut actual = db.file_text(position.file_id).text(&db).to_string();
    completions
        .into_iter()
        .exactly_one()
        .expect("more than one completion")
        .text_edit
        .apply(&mut actual);
    expect.assert_eq(&actual);
}

#[test]
fn keyword_since_edition_completes_without_raw_on_old_edition() {
    check(
        r#"
//- /a.rs crate:a edition:2015
pub fn dyn() {}

//- /b.rs crate:b edition:2015 deps:a new_source_root:local
fn foo() {
    a::dyn$0
"#,
        expect![[r#"
            fn foo() {
                a::dyn();$0
        "#]],
    );

    check(
        r#"
//- /a.rs crate:a edition:2018
pub fn r#dyn() {}

//- /b.rs crate:b edition:2015 deps:a new_source_root:local
fn foo() {
    a::dyn$0
"#,
        expect![[r#"
            fn foo() {
                a::dyn();$0
        "#]],
    );
}

#[test]
fn keyword_since_edition_completes_with_raw_on_new_edition() {
    check(
        r#"
//- /a.rs crate:a edition:2015
pub fn dyn() {}

//- /b.rs crate:b edition:2018 deps:a new_source_root:local
fn foo() {
    a::dyn$0
"#,
        expect![[r#"
            fn foo() {
                a::r#dyn();$0
        "#]],
    );

    check(
        r#"
//- /a.rs crate:a edition:2018
pub fn r#dyn() {}

//- /b.rs crate:b edition:2018 deps:a new_source_root:local
fn foo() {
    a::dyn$0
"#,
        expect![[r#"
            fn foo() {
                a::r#dyn();$0
        "#]],
    );
}
