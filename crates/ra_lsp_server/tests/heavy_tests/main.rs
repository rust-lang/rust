mod support;

use std::{
    collections::HashMap,
    time::Instant,
};

use lsp_types::{
    CodeActionContext, DocumentFormattingParams, FormattingOptions, Position, Range,
};
use ra_lsp_server::req::{
    CodeActionParams, CodeActionRequest, Formatting, Runnables, RunnablesParams, CompletionParams, Completion,
};
use serde_json::json;
use tempfile::TempDir;

use crate::support::{project, project_with_tmpdir};

const LOG: &'static str = "";

#[test]
fn completes_items_from_standard_library() {
    let project_start = Instant::now();
    let server = project(
        r#"
//- Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- src/lib.rs
use std::collections::Spam;
"#,
    );
    server.wait_for_message("workspace loaded");
    eprintln!("loading took    {:?}", project_start.elapsed());
    let completion_start = Instant::now();
    let res = server.send_request::<Completion>(CompletionParams {
        text_document: server.doc_id("src/lib.rs"),
        context: None,
        position: Position::new(0, 23),
    });
    assert!(format!("{}", res).contains("HashMap"));
    eprintln!("completion took {:?}", completion_start.elapsed());
}

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
    server.wait_for_message("workspace loaded");
    server.request::<Runnables>(
        RunnablesParams { text_document: server.doc_id("lib.rs"), position: None },
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
    server.wait_for_message("workspace loaded");
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
fn test_format_document() {
    let server = project(
        r#"
[package]
name = "foo"
version = "0.0.0"

//- src/lib.rs
mod bar;

fn main() {
}

pub use std::collections::HashMap;
"#,
    );
    server.wait_for_message("workspace loaded");

    server.request::<Formatting>(
        DocumentFormattingParams {
            text_document: server.doc_id("src/lib.rs"),
            options: FormattingOptions {
                tab_size: 4,
                insert_spaces: false,
                properties: HashMap::new(),
            },
        },
        json!([
            {
                "newText": r#"mod bar;

fn main() {}

pub use std::collections::HashMap;
"#,
                "range": {
                    "end": {
                        "character": 0,
                        "line": 6
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
    server.wait_for_message("workspace loaded");
    let empty_context = || CodeActionContext { diagnostics: Vec::new(), only: None };
    server.request::<CodeActionRequest>(
        CodeActionParams {
            text_document: server.doc_id("src/lib.rs"),
            range: Range::new(Position::new(0, 4), Position::new(0, 7)),
            context: empty_context(),
        },
        json!([
          {
            "command": {
              "arguments": [
                {
                  "cursorPosition": null,
                  "label": "create module",
                  "workspaceEdit": {
                    "documentChanges": [
                      {
                        "kind": "create",
                        "uri": "file:///[..]/src/bar.rs"
                      }
                    ]
                  }
                }
              ],
              "command": "rust-analyzer.applySourceChange",
              "title": "create module"
            },
            "title": "create module"
          }
        ]),
    );

    server.request::<CodeActionRequest>(
        CodeActionParams {
            text_document: server.doc_id("src/lib.rs"),
            range: Range::new(Position::new(2, 4), Position::new(2, 7)),
            context: empty_context(),
        },
        json!([]),
    );
}

#[test]
fn test_missing_module_code_action_in_json_project() {
    let tmp_dir = TempDir::new().unwrap();
    let code = format!(
        r#"
//- rust-project.json
{{ 
    "roots": [ "{PATH}" ], 
    "crates": [ {{ "root_module": "{PATH}/src/lib.rs", "deps": [], "edition": "2015" }} ] 
}}

//- src/lib.rs
mod bar;

fn main() {}
"#,
        PATH = tmp_dir.path().display()
    );
    let server = project_with_tmpdir(tmp_dir, &code);
    server.wait_for_message("workspace loaded");
    let empty_context = || CodeActionContext { diagnostics: Vec::new(), only: None };
    server.request::<CodeActionRequest>(
        CodeActionParams {
            text_document: server.doc_id("src/lib.rs"),
            range: Range::new(Position::new(0, 4), Position::new(0, 7)),
            context: empty_context(),
        },
        json!([
          {
            "command": {
              "arguments": [
                {
                  "cursorPosition": null,
                  "label": "create module",
                  "workspaceEdit": {
                    "documentChanges": [
                      {
                        "kind": "create",
                        "uri": "file:///[..]/src/bar.rs"
                      }
                    ]
                  }
                }
              ],
              "command": "rust-analyzer.applySourceChange",
              "title": "create module"
            },
            "title": "create module"
          }
        ]),
    );

    server.request::<CodeActionRequest>(
        CodeActionParams {
            text_document: server.doc_id("src/lib.rs"),
            range: Range::new(Position::new(2, 4), Position::new(2, 7)),
            context: empty_context(),
        },
        json!([]),
    );
}

#[test]
fn completes_items_from_second_crate_in_json_project() {
    let tmp_dir = TempDir::new().unwrap();
    let code = format!(
        r#"
//- rust-project.json
{{ 
    "roots": [ "{PATH}" ], 
    "crates": [ 
        {{ 
            "root_module": "{PATH}/foo/lib.rs",
            "deps": [
                {{
                    "name": "bar",
                    "crate": 1
                }}
            ], 
            "edition": "2015" 
        }},
        {{ "root_module": "{PATH}/bar/lib.rs", "deps": [], "edition": "2015" }}
    ] 
}}

//- bar/lib.rs
pub struct Spam;
pub struct CannedMeat;

//- foo/lib.rs
extern crate bar;
use bar::Spam;
"#,
        PATH = tmp_dir.path().display()
    );
    let server = project_with_tmpdir(tmp_dir, &code);
    server.wait_for_message("workspace loaded");
    let res = server.send_request::<Completion>(CompletionParams {
        text_document: server.doc_id("foo/lib.rs"),
        context: None,
        position: Position::new(1, 10),
    });
    assert!(format!("{}", res).contains("CannedMeat"));
}
