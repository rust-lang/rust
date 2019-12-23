mod support;

use std::{collections::HashMap, time::Instant};

use lsp_types::{
    CodeActionContext, DidOpenTextDocumentParams, DocumentFormattingParams, FormattingOptions,
    PartialResultParams, Position, Range, TextDocumentItem, TextDocumentPositionParams,
    WorkDoneProgressParams,
};
use ra_lsp_server::req::{
    CodeActionParams, CodeActionRequest, Completion, CompletionParams, DidOpenTextDocument,
    Formatting, OnEnter, Runnables, RunnablesParams,
};
use serde_json::json;
use tempfile::TempDir;
use test_utils::skip_slow_tests;

use crate::support::{project, Project};

const PROFILE: &'static str = "";
// const PROFILE: &'static str = "*@3>100";

#[test]
fn completes_items_from_standard_library() {
    if skip_slow_tests() {
        return;
    }

    let project_start = Instant::now();
    let server = Project::with_fixture(
        r#"
//- Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- src/lib.rs
use std::collections::Spam;
"#,
    )
    .with_sysroot(true)
    .server();
    server.wait_until_workspace_is_loaded();
    eprintln!("loading took    {:?}", project_start.elapsed());
    let completion_start = Instant::now();
    let res = server.send_request::<Completion>(CompletionParams {
        text_document_position: TextDocumentPositionParams::new(
            server.doc_id("src/lib.rs"),
            Position::new(0, 23),
        ),
        context: None,
        partial_result_params: PartialResultParams::default(),
        work_done_progress_params: WorkDoneProgressParams::default(),
    });
    assert!(format!("{}", res).contains("HashMap"));
    eprintln!("completion took {:?}", completion_start.elapsed());
}

#[test]
fn test_runnables_no_project() {
    if skip_slow_tests() {
        return;
    }

    let server = project(
        r"
//- lib.rs
#[test]
fn foo() {
}
",
    );
    server.wait_until_workspace_is_loaded();
    server.request::<Runnables>(
        RunnablesParams { text_document: server.doc_id("lib.rs"), position: None },
        json!([
          {
            "args": [ "test", "--", "foo", "--nocapture" ],
            "bin": "cargo",
            "env": { "RUST_BACKTRACE": "short" },
            "cwd": null,
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
            "cwd": null,
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
    if skip_slow_tests() {
        return;
    }

    let code = r#"
//- foo/Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- foo/src/lib.rs
pub fn foo() {}

//- foo/tests/spam.rs
#[test]
fn test_eggs() {}

//- bar/Cargo.toml
[package]
name = "bar"
version = "0.0.0"

//- bar/src/main.rs
fn main() {}
"#;

    let server = Project::with_fixture(code).root("foo").root("bar").server();

    server.wait_until_workspace_is_loaded();
    server.request::<Runnables>(
        RunnablesParams {
            text_document: server.doc_id("foo/tests/spam.rs"),
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
            },
            "cwd": server.path().join("foo")
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
            "cwd": server.path().join("foo"),
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
    if skip_slow_tests() {
        return;
    }

    let server = project(
        r#"
//- Cargo.toml
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
    server.wait_until_workspace_is_loaded();

    server.request::<Formatting>(
        DocumentFormattingParams {
            text_document: server.doc_id("src/lib.rs"),
            options: FormattingOptions {
                tab_size: 4,
                insert_spaces: false,
                insert_final_newline: None,
                trim_final_newlines: None,
                trim_trailing_whitespace: None,
                properties: HashMap::new(),
            },
            work_done_progress_params: WorkDoneProgressParams::default(),
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
                        "line": 7
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
fn test_format_document_2018() {
    if skip_slow_tests() {
        return;
    }

    let server = project(
        r#"
//- Cargo.toml
[package]
name = "foo"
version = "0.0.0"
edition = "2018"

//- src/lib.rs
mod bar;

async fn test() {
}

fn main() {
}

pub use std::collections::HashMap;
"#,
    );
    server.wait_until_workspace_is_loaded();

    server.request::<Formatting>(
        DocumentFormattingParams {
            text_document: server.doc_id("src/lib.rs"),
            options: FormattingOptions {
                tab_size: 4,
                insert_spaces: false,
                properties: HashMap::new(),
                insert_final_newline: None,
                trim_final_newlines: None,
                trim_trailing_whitespace: None,
            },
            work_done_progress_params: WorkDoneProgressParams::default(),
        },
        json!([
            {
                "newText": r#"mod bar;

async fn test() {}

fn main() {}

pub use std::collections::HashMap;
"#,
                "range": {
                    "end": {
                        "character": 0,
                        "line": 10
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
    if skip_slow_tests() {
        return;
    }

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
    server.wait_until_workspace_is_loaded();
    let empty_context = || CodeActionContext { diagnostics: Vec::new(), only: None };
    server.request::<CodeActionRequest>(
        CodeActionParams {
            text_document: server.doc_id("src/lib.rs"),
            range: Range::new(Position::new(0, 4), Position::new(0, 7)),
            context: empty_context(),
            partial_result_params: PartialResultParams::default(),
            work_done_progress_params: WorkDoneProgressParams::default(),
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
            partial_result_params: PartialResultParams::default(),
            work_done_progress_params: WorkDoneProgressParams::default(),
        },
        json!([]),
    );
}

#[test]
fn test_missing_module_code_action_in_json_project() {
    if skip_slow_tests() {
        return;
    }

    let tmp_dir = TempDir::new().unwrap();

    let path = tmp_dir.path();

    let project = json!({
        "roots": [path],
        "crates": [ {
            "root_module": path.join("src/lib.rs"),
            "deps": [],
            "edition": "2015",
            "atom_cfgs": [],
            "key_value_cfgs": {}
        } ]
    });

    let code = format!(
        r#"
//- rust-project.json
{PROJECT}

//- src/lib.rs
mod bar;

fn main() {{}}
"#,
        PROJECT = project.to_string(),
    );

    let server = Project::with_fixture(&code).tmp_dir(tmp_dir).server();

    server.wait_until_workspace_is_loaded();
    let empty_context = || CodeActionContext { diagnostics: Vec::new(), only: None };
    server.request::<CodeActionRequest>(
        CodeActionParams {
            text_document: server.doc_id("src/lib.rs"),
            range: Range::new(Position::new(0, 4), Position::new(0, 7)),
            context: empty_context(),
            partial_result_params: PartialResultParams::default(),
            work_done_progress_params: WorkDoneProgressParams::default(),
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
            partial_result_params: PartialResultParams::default(),
            work_done_progress_params: WorkDoneProgressParams::default(),
        },
        json!([]),
    );
}

#[test]
fn diagnostics_dont_block_typing() {
    if skip_slow_tests() {
        return;
    }

    let librs: String = (0..10).map(|i| format!("mod m{};", i)).collect();
    let libs: String = (0..10).map(|i| format!("//- src/m{}.rs\nfn foo() {{}}\n\n", i)).collect();
    let server = Project::with_fixture(&format!(
        r#"
//- Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- src/lib.rs
{}

{}

fn main() {{}}
"#,
        librs, libs
    ))
    .with_sysroot(true)
    .server();

    server.wait_until_workspace_is_loaded();
    for i in 0..10 {
        server.notification::<DidOpenTextDocument>(DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: server.doc_id(&format!("src/m{}.rs", i)).uri,
                language_id: "rust".to_string(),
                version: 0,
                text: "/// Docs\nfn foo() {}".to_string(),
            },
        });
    }
    let start = std::time::Instant::now();
    server.request::<OnEnter>(
        TextDocumentPositionParams {
            text_document: server.doc_id("src/m0.rs"),
            position: Position { line: 0, character: 5 },
        },
        json!({
          "cursorPosition": {
            "position": { "character": 4, "line": 1 },
            "textDocument": { "uri": "file:///[..]src/m0.rs" }
          },
          "label": "on enter",
          "workspaceEdit": {
            "documentChanges": [
              {
                "edits": [
                  {
                    "newText": "\n/// ",
                    "range": {
                      "end": { "character": 5, "line": 0 },
                      "start": { "character": 5, "line": 0 }
                    }
                  }
                ],
                "textDocument": { "uri": "file:///[..]src/m0.rs", "version": null }
              }
            ]
          }
        }),
    );
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 2000, "typing enter took {:?}", elapsed);
}

#[test]
fn preserves_dos_line_endings() {
    if skip_slow_tests() {
        return;
    }

    let server = Project::with_fixture(
        &"
//- Cargo.toml
[package]
name = \"foo\"
version = \"0.0.0\"

//- src/main.rs
/// Some Docs\r\nfn main() {}
",
    )
    .server();

    server.request::<OnEnter>(
        TextDocumentPositionParams {
            text_document: server.doc_id("src/main.rs"),
            position: Position { line: 0, character: 8 },
        },
        json!({
          "cursorPosition": {
            "position": { "line": 1, "character": 4 },
            "textDocument": { "uri": "file:///[..]src/main.rs" }
          },
          "label": "on enter",
          "workspaceEdit": {
            "documentChanges": [
              {
                "edits": [
                  {
                    "newText": "\r\n/// ",
                    "range": {
                      "end": { "line": 0, "character": 8 },
                      "start": { "line": 0, "character": 8 }
                    }
                  }
                ],
                "textDocument": { "uri": "file:///[..]src/main.rs", "version": null }
              }
            ]
          }
        }),
    );
}
