mod support;

use serde_json::json;
use ra_lsp_server::req::{Runnables, RunnablesParams, CodeActionRequest, CodeActionParams};
use languageserver_types::{Position, Range, CodeActionContext};

use crate::support::project;

const LOG: &'static str = "";

#[test]
fn test_runnables_no_project() {
    let server = project(
        r"
//- lib.rs
#[test]
fn foo() {
}
",
    );
    server.wait_for_feedback("workspace loaded");
    server.request::<Runnables>(
        RunnablesParams {
            text_document: server.doc_id("lib.rs"),
            position: None,
        },
        json!([
          {
            "args": [ "test", "--", "foo", "--nocapture" ],
            "bin": "cargo",
            "env": { "RUST_BACKTRACE": "short" },
            "label": "test foo",
            "range": {
              "end": { "character": 1, "line": 2 },
              "start": { "character": 0, "line": 0 }
            }
          },
          {
            "args": [
              "check",
              "--all"
            ],
            "bin": "cargo",
            "env": {},
            "label": "cargo check --all",
            "range": {
              "end": {
                "character": 0,
                "line": 0
              },
              "start": {
                "character": 0,
                "line": 0
              }
            }
          }
        ]),
    );
}

#[test]
fn test_runnables_project() {
    let server = project(
        r#"
//- Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- src/lib.rs
pub fn foo() {}

//- tests/spam.rs
#[test]
fn test_eggs() {}
"#,
    );
    server.wait_for_feedback("workspace loaded");
    server.request::<Runnables>(
        RunnablesParams {
            text_document: server.doc_id("tests/spam.rs"),
            position: None,
        },
        json!([
          {
            "args": [ "test", "--package", "foo", "--test", "spam", "--", "test_eggs", "--nocapture" ],
            "bin": "cargo",
            "env": { "RUST_BACKTRACE": "short" },
            "label": "test test_eggs",
            "range": {
              "end": { "character": 17, "line": 1 },
              "start": { "character": 0, "line": 0 }
            }
          },
          {
            "args": [
              "check",
              "--package",
              "foo",
              "--test",
              "spam"
            ],
            "bin": "cargo",
            "env": {},
            "label": "cargo check -p foo",
            "range": {
              "end": {
                "character": 0,
                "line": 0
              },
              "start": {
                "character": 0,
                "line": 0
              }
            }
          }
        ])
    );
}

#[test]
fn test_missing_module_code_action() {
    let server = project(
        r#"
//- Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- src/lib.rs
mod bar;

fn main() {}
"#,
    );
    server.wait_for_feedback("workspace loaded");
    let empty_context = || CodeActionContext {
        diagnostics: Vec::new(),
        only: None,
    };
    server.request::<CodeActionRequest>(
        CodeActionParams {
            text_document: server.doc_id("src/lib.rs"),
            range: Range::new(Position::new(0, 0), Position::new(0, 7)),
            context: empty_context(),
        },
        json!([
            {
              "arguments": [
                {
                  "cursorPosition": null,
                  "fileSystemEdits": [
                    {
                        "type": "createFile",
                        "uri": "file:///[..]/src/bar.rs"
                    }
                  ],
                  "label": "create module",
                  "sourceFileEdits": []
                }
              ],
              "command": "ra-lsp.applySourceChange",
              "title": "create module"
            }
        ]),
    );

    server.request::<CodeActionRequest>(
        CodeActionParams {
            text_document: server.doc_id("src/lib.rs"),
            range: Range::new(Position::new(2, 0), Position::new(2, 7)),
            context: empty_context(),
        },
        json!([]),
    );
}
