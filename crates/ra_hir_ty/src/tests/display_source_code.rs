use super::displayed_source_at_pos;
use crate::test_db::TestDB;
use ra_db::fixture::WithFixture;

#[test]
fn qualify_path_to_submodule() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs

mod foo {
    pub struct Foo;
}

fn bar() {
    let foo: foo::Foo = foo::Foo;
    foo<|>
}

"#,
    );
    assert_eq!("foo::Foo", displayed_source_at_pos(&db, pos));
}
