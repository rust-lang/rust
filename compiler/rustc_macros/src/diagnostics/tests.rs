use crate::diagnostics::utils::slugify;

#[test]
fn slugs() {
    assert_eq!(slugify("ExampleStructName"), "example_struct_name");
    assert_eq!(slugify("foo-bar"), "foo_bar");
}
