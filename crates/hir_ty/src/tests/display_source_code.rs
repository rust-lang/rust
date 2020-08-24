use super::check_types_source_code;

#[test]
fn qualify_path_to_submodule() {
    check_types_source_code(
        r#"
mod foo {
    pub struct Foo;
}

fn bar() {
    let foo: foo::Foo = foo::Foo;
    foo
}  //^ foo::Foo

"#,
    );
}

#[test]
fn omit_default_type_parameters() {
    check_types_source_code(
        r#"
struct Foo<T = u8> { t: T }
fn main() {
    let foo = Foo { t: 5u8 };
    foo;
}  //^ Foo
"#,
    );

    check_types_source_code(
        r#"
struct Foo<K, T = u8> { k: K, t: T }
fn main() {
    let foo = Foo { k: 400, t: 5u8 };
    foo;
}   //^ Foo<i32>
"#,
    );
}
