use std::sync::Arc;
use std::fmt::Write;
use std::path::{PathBuf, Path};
use std::fs;

use ra_db::{SyntaxDatabase, salsa::Database};
use ra_syntax::ast::{self, AstNode};
use test_utils::{project_dir, assert_eq_text, read_text};

use crate::{
    source_binder,
    mock::MockDatabase,
};

// These tests compare the inference results for all expressions in a file
// against snapshots of the expected results. If you change something and these
// tests fail expectedly, you can update the comparison files by deleting them
// and running the tests again. Similarly, to add a new test, just write the
// test here in the same pattern and it will automatically write the snapshot.

#[test]
fn infer_basics() {
    check_inference(
        r#"
fn test(a: u32, b: isize, c: !, d: &str) {
    a;
    b;
    c;
    d;
    1usize;
    1isize;
    "test";
    1.0f32;
}"#,
        "basics.txt",
    );
}

#[test]
fn infer_let() {
    check_inference(
        r#"
fn test() {
    let a = 1isize;
    let b: usize = 1;
    let c = b;
}
}"#,
        "let.txt",
    );
}

#[test]
fn infer_paths() {
    check_inference(
        r#"
fn a() -> u32 { 1 }

mod b {
    fn c() -> u32 { 1 }
}

fn test() {
    a();
    b::c();
}
}"#,
        "paths.txt",
    );
}

#[test]
fn infer_struct() {
    check_inference(
        r#"
struct A {
    b: B,
    c: C,
}
struct B;
struct C(usize);

fn test() {
    let c = C(1);
    B;
    let a: A = A { b: B, c: C(1) };
    a.b;
    a.c;
}
"#,
        "struct.txt",
    );
}

#[test]
fn infer_enum() {
    check_inference(
        r#"
enum E {
  V1 { field: u32 },
  V2
}
fn test() {
  E::V1 { field: 1 };
  E::V2;
}"#,
        "enum.txt",
    );
}

#[test]
fn infer_refs() {
    check_inference(
        r#"
fn test(a: &u32, b: &mut u32, c: *const u32, d: *mut u32) {
    a;
    *a;
    &a;
    &mut a;
    b;
    *b;
    &b;
    c;
    *c;
    d;
    *d;
}
"#,
        "refs_and_ptrs.txt",
    );
}

#[test]
fn infer_literals() {
    check_inference(
        r##"
fn test() {
    5i32;
    "hello";
    b"bytes";
    'c';
    b'b';
    3.14;
    5000;
    false;
    true;
    r#"
        //! doc
        // non-doc
        mod foo {}
        "#;
    br#"yolo"#;
}
"##,
        "literals.txt",
    );
}

#[test]
fn infer_unary_op() {
    check_inference(
        r#"
enum SomeType {}

fn test(x: SomeType) {
    let b = false;
    let c = !b;
    let a = 100;
    let d: i128 = -a;
    let e = -100;
    let f = !!!true;
    -3.14;
    -x;
    !x;
    -"hello";
}
"#,
        "unary_op.txt",
    );
}

#[test]
fn infer_backwards() {
    check_inference(
        r#"
fn takes_u32(x: u32) {}

struct S { i32_field: i32 }

fn test() -> &mut &f64 {
    let a = unknown_function();
    takes_u32(a);
    let b = unknown_function();
    S { i32_field: b };
    let c = unknown_function();
    &mut &c
}
"#,
        "backwards.txt",
    );
}

#[test]
fn infer_self() {
    check_inference(
        r#"
struct S;

impl S {
    fn test(&self) {
        self;
    }
    fn test2(self: &Self) {
        self;
    }
}
"#,
        "self.txt",
    );
}

#[test]
fn infer_binary_op() {
    check_inference(
        r#"
fn f(x: bool) -> i32 {
    0i32
}

fn test() -> bool {
    let x = a && b;
    let y = true || false;
    let z = x == y;
    let minus_forty: isize = -40isize;
    let h = minus_forty <= CONST_2;
    let c = f(z || y) + 5;
    let d = b;
    let g = minus_forty ^= i;
    let ten: usize = 10;
    let ten_is_eleven = ten == some_num;

    ten < 3
}
"#,
        "binary_op.txt",
    );
}

#[test]
fn infer_field_autoderef() {
    check_inference(
        r#"
struct A {
    b: B,
}
struct B;

fn test1(a: A) {
    let a1 = a;
    a1.b;
    let a2 = &a;
    a2.b;
    let a3 = &mut a;
    a3.b;
    let a4 = &&&&&&&a;
    a4.b;
    let a5 = &mut &&mut &&mut a;
    a5.b;
}

fn test2(a1: *const A, a2: *mut A) {
    a1.b;
    a2.b;
}
"#,
        "field_autoderef.txt",
    );
}

#[test]
fn infer_bug_484() {
    check_inference(
        r#"
fn test() {
   let x = if true {};
}
"#,
        "bug_484.txt",
    );
}

#[test]
fn infer_inherent_method() {
    check_inference(
        r#"
struct A;

impl A {
    fn foo(self, x: u32) -> i32 {}
}

mod b {
    impl super::A {
        fn bar(&self, x: u64) -> i64 {}
    }
}

fn test(a: A) {
    a.foo(1);
    (&a).bar(1);
    a.bar(1);
}
"#,
        "inherent_method.txt",
    );
}

#[test]
fn infer_tuple() {
    check_inference(
        r#"
fn test(x: &str, y: isize) {
    let a: (u32, &str) = (1, "a");
    let b = (a, x);
    let c = (y, x);
    let d = (c, x);
    let e = (1, "e");
    let f = (e, "d");
}
"#,
        "tuple.txt",
    );
}

#[test]
fn infer_array() {
    check_inference(
        r#"
fn test(x: &str, y: isize) {
    let a = [x];
    let b = [a, a];
    let c = [b, b];

    let d = [y, 1, 2, 3];
    let d = [1, y, 2, 3];
    let e = [y];
    let f = [d, d];
    let g = [e, e];

    let h = [1, 2];
    let i = ["a", "b"];

    let b = [a, ["b"]];
    let x: [u8; 0] = [];
}
"#,
        "array.txt",
    );
}

#[test]
fn infer_pattern() {
    check_inference(
        r#"
fn test(x: &i32) {
    let y = x;
    let &z = x;
    let a = z;
    let (c, d) = (1, "hello");
}
"#,
        "pattern.txt",
    );
}

fn infer(content: &str) -> String {
    let (db, _, file_id) = MockDatabase::with_single_file(content);
    let source_file = db.source_file(file_id);
    let mut acc = String::new();
    for fn_def in source_file
        .syntax()
        .descendants()
        .filter_map(ast::FnDef::cast)
    {
        let func = source_binder::function_from_source(&db, file_id, fn_def).unwrap();
        let inference_result = func.infer(&db);
        let body_syntax_mapping = func.body_syntax_mapping(&db);
        let mut types = Vec::new();
        for (pat, ty) in inference_result.type_of_pat.iter() {
            let syntax_ptr = match body_syntax_mapping.pat_syntax(pat) {
                Some(sp) => sp,
                None => continue,
            };
            types.push((syntax_ptr, ty));
        }
        for (expr, ty) in inference_result.type_of_expr.iter() {
            let syntax_ptr = match body_syntax_mapping.expr_syntax(expr) {
                Some(sp) => sp,
                None => continue,
            };
            types.push((syntax_ptr, ty));
        }
        // sort ranges for consistency
        types.sort_by_key(|(ptr, _)| (ptr.range().start(), ptr.range().end()));
        for (syntax_ptr, ty) in &types {
            let node = syntax_ptr.resolve(&source_file);
            write!(
                acc,
                "{} '{}': {}\n",
                syntax_ptr.range(),
                ellipsize(node.text().to_string().replace("\n", " "), 15),
                ty
            )
            .unwrap();
        }
    }
    acc
}

fn check_inference(content: &str, data_file: impl AsRef<Path>) {
    let data_file_path = test_data_dir().join(data_file);
    let result = infer(content);

    if !data_file_path.exists() {
        println!("File with expected result doesn't exist, creating...\n");
        println!("{}\n{}", content, result);
        fs::write(&data_file_path, &result).unwrap();
        panic!("File {:?} with expected result was created", data_file_path);
    }

    let expected = read_text(&data_file_path);
    assert_eq_text!(&expected, &result);
}

fn ellipsize(mut text: String, max_len: usize) -> String {
    if text.len() <= max_len {
        return text;
    }
    let ellipsis = "...";
    let e_len = ellipsis.len();
    let mut prefix_len = (max_len - e_len) / 2;
    while !text.is_char_boundary(prefix_len) {
        prefix_len += 1;
    }
    let mut suffix_len = max_len - e_len - prefix_len;
    while !text.is_char_boundary(text.len() - suffix_len) {
        suffix_len += 1;
    }
    text.replace_range(prefix_len..text.len() - suffix_len, ellipsis);
    text
}

fn test_data_dir() -> PathBuf {
    project_dir().join("crates/ra_hir/src/ty/tests/data")
}

#[test]
fn typing_whitespace_inside_a_function_should_not_invalidate_types() {
    let (mut db, pos) = MockDatabase::with_position(
        "
        //- /lib.rs
        fn foo() -> i32 {
            <|>1 + 1
        }
    ",
    );
    let func = source_binder::function_from_position(&db, pos).unwrap();
    {
        let events = db.log_executed(|| {
            func.infer(&db);
        });
        assert!(format!("{:?}", events).contains("infer"))
    }

    let new_text = "
        fn foo() -> i32 {
            1
            +
            1
        }
    "
    .to_string();

    db.query_mut(ra_db::FileTextQuery)
        .set(pos.file_id, Arc::new(new_text));

    {
        let events = db.log_executed(|| {
            func.infer(&db);
        });
        assert!(!format!("{:?}", events).contains("infer"), "{:#?}", events)
    }
}
