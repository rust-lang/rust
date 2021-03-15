//! proc-macro tests

#[macro_use]
mod utils;
use test_utils::assert_eq_text;
use utils::*;

#[test]
fn test_derive_serialize_proc_macro() {
    assert_expand(
        "serde_derive",
        "Serialize",
        "1.0",
        r"struct Foo {}",
        include_str!("fixtures/test_serialize_proc_macro.txt"),
    );
}

#[test]
fn test_derive_serialize_proc_macro_failed() {
    assert_expand(
        "serde_derive",
        "Serialize",
        "1.0",
        r"struct {}",
        r##"
SUBTREE $
  IDENT   compile_error 4294967295
  PUNCH   ! [alone] 4294967295
  SUBTREE {} 4294967295
    LITERAL "expected identifier" 4294967295
"##,
    );
}

#[test]
fn test_derive_proc_macro_list() {
    let res = list("serde_derive", "1").join("\n");

    assert_eq_text!(
        r#"Serialize [CustomDerive]
Deserialize [CustomDerive]"#,
        &res
    );
}

/// Tests that we find and classify non-derive macros correctly.
#[test]
fn list_test_macros() {
    let res = list("proc_macro_test", "0.0.0").join("\n");

    assert_eq_text!(
        r#"function_like_macro [FuncLike]
attribute_macro [Attr]
DummyTrait [CustomDerive]"#,
        &res
    );
}

#[test]
fn test_version_check() {
    let path = fixtures::dylib_path("proc_macro_test", "0.0.0");
    let info = proc_macro_api::read_dylib_info(&path).unwrap();
    assert!(info.version.1 >= 50);
}
