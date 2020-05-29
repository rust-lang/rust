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

#[test]
fn omit_default_type_parameters() {
    let (db, pos) = TestDB::with_position(
        r"
        //- /main.rs
        struct Foo<T = u8> { t: T }
        fn main() {
            let foo = Foo { t: 5u8 };
            foo<|>;
        }
        ",
    );
    assert_eq!("Foo", displayed_source_at_pos(&db, pos));

    let (db, pos) = TestDB::with_position(
        r"
        //- /main.rs
        struct Foo<K, T = u8> { k: K, t: T }
        fn main() {
            let foo = Foo { k: 400, t: 5u8 };
            foo<|>;
        }
        ",
    );
    assert_eq!("Foo<i32>", displayed_source_at_pos(&db, pos));
}
