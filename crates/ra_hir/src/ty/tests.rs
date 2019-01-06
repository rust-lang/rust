use std::sync::Arc;
use std::fmt::Write;
use std::path::{PathBuf, Path};
use std::fs;

use salsa::Database;

use ra_db::{SyntaxDatabase};
use ra_syntax::ast::{self, AstNode};
use test_utils::{project_dir, assert_eq_text, read_text};

use crate::{
    source_binder,
    mock::MockDatabase,
};

// These tests compare the inference results for all expressions in a file
// against snapshots of the current results. If you change something and these
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
        "0001_basics.txt",
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
        "0002_let.txt",
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
        "0003_paths.txt",
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
        "0004_struct.txt",
    );
}

#[test]
fn infer_refs_and_ptrs() {
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
        "0005_refs.txt",
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
        "0006_backwards.txt",
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
        "0007_self.txt",
    );
}

#[test]
fn infer_boolean_op() {
    check_inference(
        r#"
fn f(x: bool) -> i32 {
    0i32
}

fn test() {
    let x = a && b;
    let y = true || false;
    let z = x == y;
    let h = CONST_1 <= CONST_2;
    let c = f(z || y) + 5;
    let d = b;
    let e = 3i32 && "hello world";

    10 < 3
}
"#,
        "0008_boolean_op.txt",
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
        let func = source_binder::function_from_source(&db, file_id, fn_def)
            .unwrap()
            .unwrap();
        let inference_result = func.infer(&db).unwrap();
        for (syntax_ptr, ty) in &inference_result.type_of {
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
#[should_panic] // TODO this should work once hir::Expr is used
fn typing_whitespace_inside_a_function_should_not_invalidate_types() {
    let (mut db, pos) = MockDatabase::with_position(
        "
        //- /lib.rs
        fn foo() -> i32 {
            <|>1 + 1
        }
    ",
    );
    let func = source_binder::function_from_position(&db, pos)
        .unwrap()
        .unwrap();
    {
        let events = db.log_executed(|| {
            func.infer(&db).unwrap();
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
            func.infer(&db).unwrap();
        });
        assert!(!format!("{:?}", events).contains("infer"), "{:#?}", events)
    }
}
