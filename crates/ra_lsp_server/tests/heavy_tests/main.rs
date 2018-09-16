#[macro_use]
extern crate crossbeam_channel;
extern crate tempdir;
extern crate languageserver_types;
extern crate serde;
extern crate serde_json;
extern crate gen_lsp_server;
extern crate flexi_logger;
extern crate ra_lsp_server;

mod support;

use ra_lsp_server::req::{Runnables, RunnablesParams};

use support::project;


const LOG: &'static str = "";

#[test]
fn test_runnables_no_project() {
    let server = project(r"
//- lib.rs
#[test]
fn foo() {
}
");
    server.request::<Runnables>(
        RunnablesParams {
            text_document: server.doc_id("lib.rs"),
            position: None,
        },
        r#"[
          {
            "args": [ "test", "--", "foo", "--nocapture" ],
            "bin": "cargo",
            "env": { "RUST_BACKTRACE": "short" },
            "label": "test foo",
            "range": {
              "end": { "character": 1, "line": 2 },
              "start": { "character": 0, "line": 0 }
            }
          }
        ]"#
    );
}

#[test]
fn test_runnables_project() {
    let server = project(r#"
//- Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- src/lib.rs
pub fn foo() {}

//- tests/spam.rs
#[test]
fn test_eggs() {}
"#);
    server.wait_for_feedback("workspace loaded");
    server.request::<Runnables>(
        RunnablesParams {
            text_document: server.doc_id("tests/spam.rs"),
            position: None,
        },
        r#"[
          {
            "args": [ "test", "--package", "foo", "--test", "spam", "--", "test_eggs", "--nocapture" ],
            "bin": "cargo",
            "env": { "RUST_BACKTRACE": "short" },
            "label": "test test_eggs",
            "range": {
              "end": { "character": 17, "line": 1 },
              "start": { "character": 0, "line": 0 }
            }
          }
        ]"#
    );
}

// #[test]
// fn test_deps() {
//     let server = project(r#"
// //- Cargo.toml
// [package]
// name = "foo"
// version = "0.0.0"
// [dependencies]
// regex = "=1.0.4"

// //- src/lib.rs
// extern crate regex;
// "#);
//     server.wait_for_feedback("workspace loaded");
//     server.wait_for_feedback_n("library loaded", 9);
// }
