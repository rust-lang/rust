//! The most high-level integrated tests for rust-analyzer.
//!
//! This tests run a full LSP event loop, spawn cargo and process stdlib from
//! sysroot. For this reason, the tests here are very slow, and should be
//! avoided unless absolutely necessary.
//!
//! In particular, it's fine *not* to test that client & server agree on
//! specific JSON shapes here -- there's little value in such tests, as we can't
//! be sure without a real client anyway.

#![warn(rust_2018_idioms, unused_lifetimes)]
#![allow(clippy::disallowed_types)]

#[cfg(not(feature = "in-rust-tree"))]
mod sourcegen;
mod support;
mod testdir;
mod tidy;

use std::{collections::HashMap, path::PathBuf, sync::Once, time::Instant};

use lsp_types::{
    notification::{DidChangeTextDocument, DidOpenTextDocument, DidSaveTextDocument},
    request::{
        CodeActionRequest, Completion, Formatting, GotoTypeDefinition, HoverRequest,
        InlayHintRequest, InlayHintResolveRequest, WillRenameFiles, WorkspaceSymbolRequest,
    },
    CodeAction, CodeActionContext, CodeActionOrCommand, CodeActionParams, CodeActionResponse,
    CompletionParams, DidChangeTextDocumentParams, DidOpenTextDocumentParams,
    DidSaveTextDocumentParams, DocumentFormattingParams, FileRename, FormattingOptions,
    GotoDefinitionParams, Hover, HoverParams, InlayHint, InlayHintLabel, InlayHintParams,
    PartialResultParams, Position, Range, RenameFilesParams, TextDocumentContentChangeEvent,
    TextDocumentIdentifier, TextDocumentItem, TextDocumentPositionParams, Url,
    VersionedTextDocumentIdentifier, WorkDoneProgressParams,
};
use paths::Utf8PathBuf;
use rust_analyzer::lsp::ext::{OnEnter, Runnables, RunnablesParams, UnindexedProject};
use serde_json::json;
use stdx::format_to_acc;
use support::Server;
use test_utils::skip_slow_tests;
use testdir::TestDir;
use tracing_subscriber::fmt::TestWriter;

use crate::support::{project, Project};

#[test]
fn completes_items_from_standard_library() {
    if skip_slow_tests() {
        return;
    }

    let server = Project::with_fixture(
        r#"
//- /Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /src/lib.rs
use std::collections::Spam;
"#,
    )
    .with_config(serde_json::json!({
        "cargo": { "sysroot": "discover" },
    }))
    .server()
    .wait_until_workspace_is_loaded();

    let res = server.send_request::<Completion>(CompletionParams {
        text_document_position: TextDocumentPositionParams::new(
            server.doc_id("src/lib.rs"),
            Position::new(0, 23),
        ),
        context: None,
        partial_result_params: PartialResultParams::default(),
        work_done_progress_params: WorkDoneProgressParams::default(),
    });
    assert!(res.to_string().contains("HashMap"));
}

#[test]
fn resolves_inlay_hints() {
    if skip_slow_tests() {
        return;
    }

    let server = Project::with_fixture(
        r#"
//- /Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /src/lib.rs
struct Foo;
fn f() {
    let x = Foo;
}
"#,
    )
    .server()
    .wait_until_workspace_is_loaded();

    let res = server.send_request::<InlayHintRequest>(InlayHintParams {
        range: Range::new(Position::new(0, 0), Position::new(3, 1)),
        text_document: server.doc_id("src/lib.rs"),
        work_done_progress_params: WorkDoneProgressParams::default(),
    });
    let mut hints = serde_json::from_value::<Option<Vec<InlayHint>>>(res).unwrap().unwrap();
    let hint = hints.pop().unwrap();
    assert!(hint.data.is_some());
    assert!(
        matches!(&hint.label, InlayHintLabel::LabelParts(parts) if parts[1].location.is_none())
    );
    let res = server.send_request::<InlayHintResolveRequest>(hint);
    let hint = serde_json::from_value::<InlayHint>(res).unwrap();
    assert!(hint.data.is_none());
    assert!(
        matches!(&hint.label, InlayHintLabel::LabelParts(parts) if parts[1].location.is_some())
    );
}

#[test]
fn completes_items_from_standard_library_in_cargo_script() {
    // this test requires nightly so CI can't run it
    if skip_slow_tests() || std::env::var("CI").is_ok() {
        return;
    }

    let server = Project::with_fixture(
        r#"
//- /dependency/Cargo.toml
[package]
name = "dependency"
version = "0.1.0"
//- /dependency/src/lib.rs
pub struct SpecialHashMap;
//- /dependency2/Cargo.toml
[package]
name = "dependency2"
version = "0.1.0"
//- /dependency2/src/lib.rs
pub struct SpecialHashMap2;
//- /src/lib.rs
#!/usr/bin/env -S cargo +nightly -Zscript
---
[dependencies]
dependency = { path = "../dependency" }
---
use dependency::Spam;
use dependency2::Spam;
"#,
    )
    .with_config(serde_json::json!({
        "cargo": { "sysroot": null },
        "linkedProjects": ["src/lib.rs"],
    }))
    .server()
    .wait_until_workspace_is_loaded();

    let res = server.send_request::<Completion>(CompletionParams {
        text_document_position: TextDocumentPositionParams::new(
            server.doc_id("src/lib.rs"),
            Position::new(5, 18),
        ),
        context: None,
        partial_result_params: PartialResultParams::default(),
        work_done_progress_params: WorkDoneProgressParams::default(),
    });
    assert!(res.to_string().contains("SpecialHashMap"), "{}", res.to_string());

    let res = server.send_request::<Completion>(CompletionParams {
        text_document_position: TextDocumentPositionParams::new(
            server.doc_id("src/lib.rs"),
            Position::new(6, 18),
        ),
        context: None,
        partial_result_params: PartialResultParams::default(),
        work_done_progress_params: WorkDoneProgressParams::default(),
    });
    assert!(!res.to_string().contains("SpecialHashMap"));

    server.write_file_and_save(
        "src/lib.rs",
        r#"#!/usr/bin/env -S cargo +nightly -Zscript
---
[dependencies]
dependency2 = { path = "../dependency2" }
---
use dependency::Spam;
use dependency2::Spam;
"#
        .to_owned(),
    );

    let server = server.wait_until_workspace_is_loaded();

    std::thread::sleep(std::time::Duration::from_secs(3));

    let res = server.send_request::<Completion>(CompletionParams {
        text_document_position: TextDocumentPositionParams::new(
            server.doc_id("src/lib.rs"),
            Position::new(5, 18),
        ),
        context: None,
        partial_result_params: PartialResultParams::default(),
        work_done_progress_params: WorkDoneProgressParams::default(),
    });
    assert!(!res.to_string().contains("SpecialHashMap"));

    let res = server.send_request::<Completion>(CompletionParams {
        text_document_position: TextDocumentPositionParams::new(
            server.doc_id("src/lib.rs"),
            Position::new(6, 18),
        ),
        context: None,
        partial_result_params: PartialResultParams::default(),
        work_done_progress_params: WorkDoneProgressParams::default(),
    });
    assert!(res.to_string().contains("SpecialHashMap"));
}

#[test]
fn test_runnables_project() {
    if skip_slow_tests() {
        return;
    }

    let server = Project::with_fixture(
        r#"
//- /foo/Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /foo/src/lib.rs
pub fn foo() {}

//- /foo/tests/spam.rs
#[test]
fn test_eggs() {}

//- /bar/Cargo.toml
[package]
name = "bar"
version = "0.0.0"

//- /bar/src/main.rs
fn main() {}
"#,
    )
    .root("foo")
    .root("bar")
    .server()
    .wait_until_workspace_is_loaded();

    server.request::<Runnables>(
        RunnablesParams { text_document: server.doc_id("foo/tests/spam.rs"), position: None },
        json!([
          {
            "args": {
              "cargoArgs": ["test", "--package", "foo", "--test", "spam"],
              "executableArgs": ["test_eggs", "--exact", "--show-output"],
              "cargoExtraArgs": [],
              "overrideCargo": null,
              "cwd": server.path().join("foo"),
              "workspaceRoot": server.path().join("foo")
            },
            "kind": "cargo",
            "label": "test test_eggs",
            "location": {
              "targetRange": {
                "end": { "character": 17, "line": 1 },
                "start": { "character": 0, "line": 0 }
              },
              "targetSelectionRange": {
                "end": { "character": 12, "line": 1 },
                "start": { "character": 3, "line": 1 }
              },
              "targetUri": "file:///[..]/tests/spam.rs"
            }
          },
          {
            "args": {
              "overrideCargo": null,
              "cwd": server.path().join("foo"),
              "workspaceRoot": server.path().join("foo"),
              "cargoArgs": [
                "test",
                "--package",
                "foo",
                "--test",
                "spam"
              ],
              "cargoExtraArgs": [],
              "executableArgs": [
                "",
                "--show-output"
              ]
            },
            "kind": "cargo",
            "label": "test-mod ",
            "location": {
              "targetUri": "file:///[..]/tests/spam.rs",
              "targetRange": {
                "start": {
                  "line": 0,
                  "character": 0
                },
                "end": {
                  "line": 3,
                  "character": 0
                }
              },
              "targetSelectionRange": {
                "start": {
                  "line": 0,
                  "character": 0
                },
                "end": {
                  "line": 3,
                  "character": 0
                }
              }
            },
          },
          {
            "args": {
              "cargoArgs": ["check", "--package", "foo", "--all-targets"],
              "executableArgs": [],
              "cargoExtraArgs": [],
              "overrideCargo": null,
              "cwd": server.path().join("foo"),
              "workspaceRoot": server.path().join("foo")
            },
            "kind": "cargo",
            "label": "cargo check -p foo --all-targets"
          },
          {
            "args": {
              "cargoArgs": ["test", "--package", "foo", "--all-targets"],
              "executableArgs": [],
              "cargoExtraArgs": [],
              "overrideCargo": null,
              "cwd": server.path().join("foo"),
              "workspaceRoot": server.path().join("foo")
            },
            "kind": "cargo",
            "label": "cargo test -p foo --all-targets"
          }
        ]),
    );
}

// Each package in these workspaces should be run from its own root
#[test]
fn test_path_dependency_runnables() {
    if skip_slow_tests() {
        return;
    }

    let server = Project::with_fixture(
        r#"
//- /consumer/Cargo.toml
[package]
name = "consumer"
version = "0.1.0"
[dependencies]
dependency = { path = "../dependency" }

//- /consumer/src/lib.rs
#[cfg(test)]
mod tests {
    #[test]
    fn consumer() {}
}

//- /dependency/Cargo.toml
[package]
name = "dependency"
version = "0.1.0"
[dev-dependencies]
devdependency = { path = "../devdependency" }

//- /dependency/src/lib.rs
#[cfg(test)]
mod tests {
    #[test]
    fn dependency() {}
}

//- /devdependency/Cargo.toml
[package]
name = "devdependency"
version = "0.1.0"

//- /devdependency/src/lib.rs
#[cfg(test)]
mod tests {
    #[test]
    fn devdependency() {}
}
        "#,
    )
    .root("consumer")
    .root("dependency")
    .root("devdependency")
    .server()
    .wait_until_workspace_is_loaded();

    for runnable in ["consumer", "dependency", "devdependency"] {
        server.request::<Runnables>(
            RunnablesParams {
                text_document: server.doc_id(&format!("{runnable}/src/lib.rs")),
                position: None,
            },
            json!([
                "{...}",
                {
                    "label": "cargo test -p [..] --all-targets",
                    "kind": "cargo",
                    "args": {
                        "overrideCargo": null,
                        "workspaceRoot": server.path().join(runnable),
                        "cwd": server.path().join(runnable),
                        "cargoArgs": [
                            "test",
                            "--package",
                            runnable,
                            "--all-targets"
                        ],
                        "cargoExtraArgs": [],
                        "executableArgs": []
                    },
                },
                "{...}",
                "{...}"
            ]),
        );
    }
}

// The main fn in packages should be run from the workspace root
#[test]
fn test_runnables_cwd() {
    if skip_slow_tests() {
        return;
    }

    let server = Project::with_fixture(
        r#"
//- /foo/Cargo.toml
[workspace]
members = ["mainpkg", "otherpkg"]

//- /foo/mainpkg/Cargo.toml
[package]
name = "mainpkg"
version = "0.1.0"

//- /foo/mainpkg/src/main.rs
fn main() {}

//- /foo/otherpkg/Cargo.toml
[package]
name = "otherpkg"
version = "0.1.0"

//- /foo/otherpkg/src/lib.rs
#[test]
fn otherpkg() {}
"#,
    )
    .root("foo")
    .server()
    .wait_until_workspace_is_loaded();

    server.request::<Runnables>(
        RunnablesParams { text_document: server.doc_id("foo/mainpkg/src/main.rs"), position: None },
        json!([
            "{...}",
            {
                "label": "cargo test -p mainpkg --all-targets",
                "kind": "cargo",
                "args": {
                    "overrideCargo": null,
                    "workspaceRoot": server.path().join("foo"),
                    "cwd": server.path().join("foo"),
                    "cargoArgs": [
                        "test",
                        "--package",
                        "mainpkg",
                        "--all-targets"
                    ],
                    "cargoExtraArgs": [],
                    "executableArgs": []
                },
            },
            "{...}",
            "{...}"
        ]),
    );

    server.request::<Runnables>(
        RunnablesParams { text_document: server.doc_id("foo/otherpkg/src/lib.rs"), position: None },
        json!([
            "{...}",
            {
                "label": "cargo test -p otherpkg --all-targets",
                "kind": "cargo",
                "args": {
                    "overrideCargo": null,
                    "workspaceRoot": server.path().join("foo"),
                    "cwd": server.path().join("foo").join("otherpkg"),
                    "cargoArgs": [
                        "test",
                        "--package",
                        "otherpkg",
                        "--all-targets"
                    ],
                    "cargoExtraArgs": [],
                    "executableArgs": []
                },
            },
            "{...}",
            "{...}"
        ]),
    );
}

#[test]
fn test_format_document() {
    if skip_slow_tests() {
        return;
    }

    let server = project(
        r#"
//- /Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /src/lib.rs
mod bar;

fn main() {
}

pub use std::collections::HashMap;
"#,
    )
    .wait_until_workspace_is_loaded();

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
                "newText": "",
                "range": {
                    "end": { "character": 0, "line": 3 },
                    "start": { "character": 11, "line": 2 }
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
//- /Cargo.toml
[package]
name = "foo"
version = "0.0.0"
edition = "2018"

//- /src/lib.rs
mod bar;

async fn test() {
}

fn main() {
}

pub use std::collections::HashMap;
"#,
    )
    .wait_until_workspace_is_loaded();

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
                "newText": "",
                "range": {
                    "end": { "character": 0, "line": 3 },
                    "start": { "character": 17, "line": 2 }
                }
            },
            {
                "newText": "",
                "range": {
                    "end": { "character": 0, "line": 6 },
                    "start": { "character": 11, "line": 5 }
                }
            }
        ]),
    );
}

#[test]
fn test_format_document_unchanged() {
    if skip_slow_tests() {
        return;
    }

    let server = project(
        r#"
//- /Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /src/lib.rs
fn main() {}
"#,
    )
    .wait_until_workspace_is_loaded();

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
        json!(null),
    );
}

#[test]
fn test_missing_module_code_action() {
    if skip_slow_tests() {
        return;
    }

    let server = project(
        r#"
//- /Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /src/lib.rs
mod bar;

fn main() {}
"#,
    )
    .wait_until_workspace_is_loaded();

    server.request::<CodeActionRequest>(
        CodeActionParams {
            text_document: server.doc_id("src/lib.rs"),
            range: Range::new(Position::new(0, 4), Position::new(0, 7)),
            context: CodeActionContext::default(),
            partial_result_params: PartialResultParams::default(),
            work_done_progress_params: WorkDoneProgressParams::default(),
        },
        json!([
            {
                "title": "Create module at `bar.rs`",
                "kind": "quickfix",
                "edit": {
                "documentChanges": [
                    {
                    "kind": "create",
                    "uri": "file://[..]/src/bar.rs"
                    }
                ]
                }
            },
            {
                "title": "Create module at `bar/mod.rs`",
                "kind": "quickfix",
                "edit": {
                "documentChanges": [
                    {
                    "kind": "create",
                    "uri": "file://[..]src/bar/mod.rs"
                    }
                ]
                }
            }
        ]),
    );

    server.request::<CodeActionRequest>(
        CodeActionParams {
            text_document: server.doc_id("src/lib.rs"),
            range: Range::new(Position::new(2, 8), Position::new(2, 8)),
            context: CodeActionContext::default(),
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

    let tmp_dir = TestDir::new();

    let path = tmp_dir.path();

    let project = json!({
        "roots": [path],
        "crates": [ {
            "root_module": path.join("src/lib.rs"),
            "deps": [],
            "edition": "2015",
            "cfg": [ "cfg_atom_1", "feature=\"cfg_1\""],
        } ]
    });

    let code = format!(
        r#"
//- /rust-project.json
{project}

//- /src/lib.rs
mod bar;

fn main() {{}}
"#,
    );

    let server =
        Project::with_fixture(&code).tmp_dir(tmp_dir).server().wait_until_workspace_is_loaded();

    server.request::<CodeActionRequest>(
        CodeActionParams {
            text_document: server.doc_id("src/lib.rs"),
            range: Range::new(Position::new(0, 4), Position::new(0, 7)),
            context: CodeActionContext::default(),
            partial_result_params: PartialResultParams::default(),
            work_done_progress_params: WorkDoneProgressParams::default(),
        },
        json!([
            {
                "title": "Create module at `bar.rs`",
                "kind": "quickfix",
                "edit": {
                "documentChanges": [
                    {
                    "kind": "create",
                    "uri": "file://[..]/src/bar.rs"
                    }
                ]
                }
            },
            {
                "title": "Create module at `bar/mod.rs`",
                "kind": "quickfix",
                "edit": {
                "documentChanges": [
                    {
                    "kind": "create",
                    "uri": "file://[..]src/bar/mod.rs"
                    }
                ]
                }
            }
        ]),
    );

    server.request::<CodeActionRequest>(
        CodeActionParams {
            text_document: server.doc_id("src/lib.rs"),
            range: Range::new(Position::new(2, 8), Position::new(2, 8)),
            context: CodeActionContext::default(),
            partial_result_params: PartialResultParams::default(),
            work_done_progress_params: WorkDoneProgressParams::default(),
        },
        json!([]),
    );
}

#[test]
fn test_opening_a_file_outside_of_indexed_workspace() {
    if skip_slow_tests() {
        return;
    }

    let tmp_dir = TestDir::new();
    let path = tmp_dir.path();

    let project = json!({
        "roots": [path],
        "crates": [ {
            "root_module": path.join("src/crate_one/lib.rs"),
            "deps": [],
            "edition": "2015",
            "cfg": [ "cfg_atom_1", "feature=\"cfg_1\""],
        } ]
    });

    let code = format!(
        r#"
//- /rust-project.json
{project}

//- /src/crate_one/lib.rs
mod bar;

fn main() {{}}
"#,
    );

    let server = Project::with_fixture(&code)
        .tmp_dir(tmp_dir)
        .with_config(serde_json::json!({
            "notifications": {
                "unindexedProject": true
            },
        }))
        .server()
        .wait_until_workspace_is_loaded();

    let uri = server.doc_id("src/crate_two/lib.rs").uri;
    server.notification::<DidOpenTextDocument>(DidOpenTextDocumentParams {
        text_document: TextDocumentItem {
            uri: uri.clone(),
            language_id: "rust".to_owned(),
            version: 0,
            text: "/// Docs\nfn foo() {}".to_owned(),
        },
    });
    let expected = json!({
        "textDocuments": [
            {
                "uri": uri
            }
        ]
    });
    server.expect_notification::<UnindexedProject>(expected);
}

#[test]
fn diagnostics_dont_block_typing() {
    if skip_slow_tests() {
        return;
    }

    let librs: String = (0..10).fold(String::new(), |mut acc, i| format_to_acc!(acc, "mod m{i};"));
    let libs: String = (0..10).fold(String::new(), |mut acc, i| {
        format_to_acc!(acc, "//- /src/m{i}.rs\nfn foo() {{}}\n\n")
    });
    let server = Project::with_fixture(&format!(
        r#"
//- /Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /src/lib.rs
{librs}

{libs}

fn main() {{}}
"#
    ))
    .with_config(serde_json::json!({
        "cargo": { "sysroot": "discover" },
    }))
    .server()
    .wait_until_workspace_is_loaded();

    for i in 0..10 {
        server.notification::<DidOpenTextDocument>(DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: server.doc_id(&format!("src/m{i}.rs")).uri,
                language_id: "rust".to_owned(),
                version: 0,
                text: "/// Docs\nfn foo() {}".to_owned(),
            },
        });
    }
    let start = Instant::now();
    server.request::<OnEnter>(
        TextDocumentPositionParams {
            text_document: server.doc_id("src/m0.rs"),
            position: Position { line: 0, character: 5 },
        },
        json!([{
            "insertTextFormat": 2,
            "newText": "\n/// $0",
            "range": {
            "end": { "character": 5, "line": 0 },
            "start": { "character": 5, "line": 0 }
            }
        }]),
    );
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 2000, "typing enter took {elapsed:?}");
}

#[test]
fn preserves_dos_line_endings() {
    if skip_slow_tests() {
        return;
    }

    let server = Project::with_fixture(
        "
//- /Cargo.toml
[package]
name = \"foo\"
version = \"0.0.0\"

//- /src/main.rs
/// Some Docs\r\nfn main() {}
",
    )
    .server()
    .wait_until_workspace_is_loaded();

    server.request::<OnEnter>(
        TextDocumentPositionParams {
            text_document: server.doc_id("src/main.rs"),
            position: Position { line: 0, character: 8 },
        },
        json!([{
            "insertTextFormat": 2,
            "newText": "\r\n/// $0",
            "range": {
            "end": { "line": 0, "character": 8 },
            "start": { "line": 0, "character": 8 }
            }
        }]),
    );
}

fn out_dirs_check_impl(root_contains_symlink: bool) {
    if skip_slow_tests() {
        return;
    }

    let mut server = Project::with_fixture(
        r###"
//- /Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /build.rs
use std::{env, fs, path::Path};

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("hello.rs");
    fs::write(
        &dest_path,
        r#"pub fn message() -> &'static str { "Hello, World!" }"#,
    )
    .unwrap();
    println!("cargo:rustc-cfg=atom_cfg");
    println!("cargo:rustc-cfg=featlike=\"set\"");
    println!("cargo:rerun-if-changed=build.rs");
}
//- /src/main.rs
#![allow(warnings)]
#![feature(rustc_attrs)]
#[rustc_builtin_macro] macro_rules! include {
    ($file:expr $(,)?) => {{ /* compiler built-in */ }};
}
#[rustc_builtin_macro] macro_rules! include_str {
    ($file:expr $(,)?) => {{ /* compiler built-in */ }};
}
#[rustc_builtin_macro] macro_rules! concat {
    ($($e:ident),+ $(,)?) => {{ /* compiler built-in */ }};
}
#[rustc_builtin_macro] macro_rules! env {
    ($name:expr $(,)?) => {{ /* compiler built-in */ }};
    ($name:expr, $error_msg:expr $(,)?) => {{ /* compiler built-in */ }};
}

include!(concat!(env!("OUT_DIR"), "/hello.rs"));

#[cfg(atom_cfg)]
struct A;
#[cfg(bad_atom_cfg)]
struct A;
#[cfg(featlike = "set")]
struct B;
#[cfg(featlike = "not_set")]
struct B;

fn main() {
    let va = A;
    let vb = B;
    let should_be_str = message();
    let another_str = include_str!("main.rs");
}
"###,
    );

    if root_contains_symlink {
        server = server.with_root_dir_contains_symlink();
    }

    let server = server
        .with_config(serde_json::json!({
            "cargo": {
                "buildScripts": {
                    "enable": true
                },
                "sysroot": null,
                "extraEnv": {
                    "RUSTC_BOOTSTRAP": "1"
                }
            }
        }))
        .server()
        .wait_until_workspace_is_loaded();

    let res = server.send_request::<HoverRequest>(HoverParams {
        text_document_position_params: TextDocumentPositionParams::new(
            server.doc_id("src/main.rs"),
            Position::new(30, 10),
        ),
        work_done_progress_params: Default::default(),
    });
    assert!(res.to_string().contains("&str"));

    let res = server.send_request::<HoverRequest>(HoverParams {
        text_document_position_params: TextDocumentPositionParams::new(
            server.doc_id("src/main.rs"),
            Position::new(31, 10),
        ),
        work_done_progress_params: Default::default(),
    });
    assert!(res.to_string().contains("&str"));

    server.request::<GotoTypeDefinition>(
        GotoDefinitionParams {
            text_document_position_params: TextDocumentPositionParams::new(
                server.doc_id("src/main.rs"),
                Position::new(28, 9),
            ),
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        },
        json!([{
            "originSelectionRange": {
                "end": { "character": 10, "line": 28 },
                "start": { "character": 8, "line": 28 }
            },
            "targetRange": {
                "end": { "character": 9, "line": 19 },
                "start": { "character": 0, "line": 18 }
            },
            "targetSelectionRange": {
                "end": { "character": 8, "line": 19 },
                "start": { "character": 7, "line": 19 }
            },
            "targetUri": "file:///[..]src/main.rs"
        }]),
    );

    server.request::<GotoTypeDefinition>(
        GotoDefinitionParams {
            text_document_position_params: TextDocumentPositionParams::new(
                server.doc_id("src/main.rs"),
                Position::new(29, 9),
            ),
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        },
        json!([{
            "originSelectionRange": {
                "end": { "character": 10, "line": 29 },
                "start": { "character": 8, "line": 29 }
            },
            "targetRange": {
                "end": { "character": 9, "line": 23 },
                "start": { "character": 0, "line": 22 }
            },
            "targetSelectionRange": {
                "end": { "character": 8, "line": 23 },
                "start": { "character": 7, "line": 23 }
            },
            "targetUri": "file:///[..]src/main.rs"
        }]),
    );
}

#[test]
fn out_dirs_check() {
    out_dirs_check_impl(false);
}

#[test]
#[cfg(not(windows))] // windows requires elevated permissions to create symlinks
fn root_contains_symlink_out_dirs_check() {
    out_dirs_check_impl(true);
}

#[test]
#[cfg(any(feature = "sysroot-abi", rust_analyzer))]
fn resolve_proc_macro() {
    use expect_test::expect;
    use vfs::AbsPathBuf;
    if skip_slow_tests() {
        return;
    }

    let sysroot = project_model::Sysroot::discover(
        &AbsPathBuf::assert_utf8(std::env::current_dir().unwrap()),
        &Default::default(),
        false,
    );

    let proc_macro_server_path = sysroot.discover_proc_macro_srv().unwrap();

    let server = Project::with_fixture(
        r###"
//- /foo/Cargo.toml
[package]
name = "foo"
version = "0.0.0"
edition = "2021"
[dependencies]
bar = {path = "../bar"}

//- /foo/src/main.rs
#![allow(internal_features)]
#![feature(rustc_attrs, decl_macro)]
use bar::Bar;

#[rustc_builtin_macro]
macro derive($item:item) {}
trait Bar {
  fn bar();
}
#[derive(Bar)]
struct Foo {}
fn main() {
  Foo::bar();
}

//- /bar/Cargo.toml
[package]
name = "bar"
version = "0.0.0"
edition = "2021"

[lib]
proc-macro = true

//- /bar/src/lib.rs
extern crate proc_macro;
use proc_macro::{Delimiter, Group, Ident, Span, TokenStream, TokenTree};
macro_rules! t {
    ($n:literal) => {
        TokenTree::from(Ident::new($n, Span::call_site()))
    };
    ({}) => {
        TokenTree::from(Group::new(Delimiter::Brace, TokenStream::new()))
    };
    (()) => {
        TokenTree::from(Group::new(Delimiter::Parenthesis, TokenStream::new()))
    };
}
#[proc_macro_derive(Bar)]
pub fn foo(_input: TokenStream) -> TokenStream {
    // We hard code the output here for preventing to use any deps
    let mut res = TokenStream::new();

    // ill behaved proc-macro will use the stdout
    // we should ignore it
    println!("I am bad guy");

    // impl Bar for Foo { fn bar() {} }
    let mut tokens = vec![t!("impl"), t!("Bar"), t!("for"), t!("Foo")];
    let mut fn_stream = TokenStream::new();
    fn_stream.extend(vec![t!("fn"), t!("bar"), t!(()), t!({})]);
    tokens.push(Group::new(Delimiter::Brace, fn_stream).into());
    res.extend(tokens);
    res
}

"###,
    )
    .with_config(serde_json::json!({
        "cargo": {
            "buildScripts": {
                "enable": true
            },
            "sysroot": null,
        },
        "procMacro": {
            "enable": true,
            "server": proc_macro_server_path.as_path().as_str(),
        }
    }))
    .root("foo")
    .root("bar")
    .server()
    .wait_until_workspace_is_loaded();

    let res = server.send_request::<HoverRequest>(HoverParams {
        text_document_position_params: TextDocumentPositionParams::new(
            server.doc_id("foo/src/main.rs"),
            Position::new(12, 9),
        ),
        work_done_progress_params: Default::default(),
    });
    let value = res.get("contents").unwrap().get("value").unwrap().as_str().unwrap();

    expect![[r#"

        ```rust
        foo::Foo
        ```

        ```rust
        fn bar()
        ```"#]]
    .assert_eq(value);
}

#[test]
fn test_will_rename_files_same_level() {
    if skip_slow_tests() {
        return;
    }

    let tmp_dir = TestDir::new();
    let tmp_dir_path = tmp_dir.path().to_owned();
    let tmp_dir_str = tmp_dir_path.as_str();
    let base_path = PathBuf::from(format!("file://{tmp_dir_str}"));

    let code = r#"
//- /Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /src/lib.rs
mod old_file;
mod from_mod;
mod to_mod;
mod old_folder;
fn main() {}

//- /src/old_file.rs

//- /src/old_folder/mod.rs
mod nested;

//- /src/old_folder/nested.rs
struct foo;
use crate::old_folder::nested::foo as bar;

//- /src/from_mod/mod.rs

//- /src/to_mod/foo.rs

"#;
    let server =
        Project::with_fixture(code).tmp_dir(tmp_dir).server().wait_until_workspace_is_loaded();

    //rename same level file
    server.request::<WillRenameFiles>(
        RenameFilesParams {
            files: vec![FileRename {
                old_uri: base_path.join("src/old_file.rs").to_str().unwrap().to_owned(),
                new_uri: base_path.join("src/new_file.rs").to_str().unwrap().to_owned(),
            }],
        },
        json!({
          "documentChanges": [
            {
              "textDocument": {
                "uri": format!("file://{}", tmp_dir_path.join("src").join("lib.rs").as_str().to_owned().replace("C:\\", "/c:/").replace('\\', "/")),
                "version": null
              },
              "edits": [
                {
                  "range": {
                    "start": {
                      "line": 0,
                      "character": 4
                    },
                    "end": {
                      "line": 0,
                      "character": 12
                    }
                  },
                  "newText": "new_file"
                }
              ]
            }
          ]
        }),
    );

    //rename file from mod.rs to foo.rs
    server.request::<WillRenameFiles>(
        RenameFilesParams {
            files: vec![FileRename {
                old_uri: base_path.join("src/from_mod/mod.rs").to_str().unwrap().to_owned(),
                new_uri: base_path.join("src/from_mod/foo.rs").to_str().unwrap().to_owned(),
            }],
        },
        json!(null),
    );

    //rename file from foo.rs to mod.rs
    server.request::<WillRenameFiles>(
        RenameFilesParams {
            files: vec![FileRename {
                old_uri: base_path.join("src/to_mod/foo.rs").to_str().unwrap().to_owned(),
                new_uri: base_path.join("src/to_mod/mod.rs").to_str().unwrap().to_owned(),
            }],
        },
        json!(null),
    );

    //rename same level file
    server.request::<WillRenameFiles>(
        RenameFilesParams {
            files: vec![FileRename {
                old_uri: base_path.join("src/old_folder").to_str().unwrap().to_owned(),
                new_uri: base_path.join("src/new_folder").to_str().unwrap().to_owned(),
            }],
        },
        json!({
          "documentChanges": [
            {
              "textDocument": {
                "uri": format!("file://{}", tmp_dir_path.join("src").join("lib.rs").as_str().to_owned().replace("C:\\", "/c:/").replace('\\', "/")),
                "version": null
              },
              "edits": [
                {
                  "range": {
                    "start": {
                      "line": 3,
                      "character": 4
                    },
                    "end": {
                      "line": 3,
                      "character": 14
                    }
                  },
                  "newText": "new_folder"
                }
              ]
            },
            {
              "textDocument": {
                "uri": format!("file://{}", tmp_dir_path.join("src").join("old_folder").join("nested.rs").as_str().to_owned().replace("C:\\", "/c:/").replace('\\', "/")),
                "version": null
              },
              "edits": [
                {
                  "range": {
                    "start": {
                      "line": 1,
                      "character": 11
                    },
                    "end": {
                      "line": 1,
                      "character": 21
                    }
                  },
                  "newText": "new_folder"
                }
              ]
            }
          ]
        }),
    );
}

#[test]
fn test_exclude_config_works() {
    if skip_slow_tests() {
        return;
    }

    let server = Project::with_fixture(
        r#"
//- /foo/Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- /foo/src/lib.rs
pub fn foo() {}

//- /bar/Cargo.toml
[package]
name = "bar"
version = "0.0.0"

//- /bar/src/lib.rs
pub fn bar() {}

//- /baz/Cargo.toml
[package]
name = "baz"
version = "0.0.0"

//- /baz/src/lib.rs
"#,
    )
    .root("foo")
    .root("bar")
    .root("baz")
    .with_config(json!({
       "files": {
           "excludeDirs": ["foo", "bar"]
        }
    }))
    .server()
    .wait_until_workspace_is_loaded();

    server.request::<WorkspaceSymbolRequest>(Default::default(), json!([]));
}

enum QueryType {
    AssistEmitMustUse,
    /// A query whose config key is a part of the global configs, so that
    /// testing for changes to this config means testing if global changes
    /// take affect.
    GlobalHover,
}

struct RatomlTest {
    urls: Vec<Url>,
    server: Server,
    tmp_path: Utf8PathBuf,
    user_config_dir: Utf8PathBuf,
}

impl RatomlTest {
    const EMIT_MUST_USE: &'static str = r#"assist.emitMustUse = true"#;
    const EMIT_MUST_NOT_USE: &'static str = r#"assist.emitMustUse = false"#;
    const EMIT_MUST_USE_SNIPPET: &'static str = r#"

impl Value {
    #[must_use]
    fn as_text(&self) -> Option<&String> {
        if let Self::Text(v) = self {
            Some(v)
        } else {
            None
        }
    }
}"#;

    const GLOBAL_TRAIT_ASSOC_ITEMS_ZERO: &'static str = r#"hover.show.traitAssocItems = 0"#;
    const GLOBAL_TRAIT_ASSOC_ITEMS_SNIPPET: &'static str = r#"
```rust
p1
```

```rust
trait RandomTrait {
    type B;
    fn abc() -> i32;
    fn def() -> i64;
}
```"#;

    fn new(
        fixtures: Vec<&str>,
        roots: Vec<&str>,
        client_config: Option<serde_json::Value>,
    ) -> Self {
        // setup();
        let tmp_dir = TestDir::new();
        let tmp_path = tmp_dir.path().to_owned();

        let full_fixture = fixtures.join("\n");

        let user_cnf_dir = TestDir::new();
        let user_config_dir = user_cnf_dir.path().to_owned();

        let mut project =
            Project::with_fixture(&full_fixture).tmp_dir(tmp_dir).user_config_dir(user_cnf_dir);

        for root in roots {
            project = project.root(root);
        }

        if let Some(client_config) = client_config {
            project = project.with_config(client_config);
        }

        let server = project.server().wait_until_workspace_is_loaded();

        let mut case = Self { urls: vec![], server, tmp_path, user_config_dir };
        let urls = fixtures.iter().map(|fixture| case.fixture_path(fixture)).collect::<Vec<_>>();
        case.urls = urls;
        case
    }

    fn fixture_path(&self, fixture: &str) -> Url {
        let mut lines = fixture.trim().split('\n');

        let mut path =
            lines.next().expect("All files in a fixture are expected to have at least one line.");

        if path.starts_with("//- minicore") {
            path = lines.next().expect("A minicore line must be followed by a path.")
        }

        path = path.strip_prefix("//- ").expect("Path must be preceded by a //- prefix ");

        let spl = path[1..].split('/');
        let mut path = self.tmp_path.clone();

        let mut spl = spl.into_iter();
        if let Some(first) = spl.next() {
            if first == "$$CONFIG_DIR$$" {
                path = self.user_config_dir.clone();
            } else {
                path = path.join(first);
            }
        }
        for piece in spl {
            path = path.join(piece);
        }

        Url::parse(
            format!(
                "file://{}",
                path.into_string().to_owned().replace("C:\\", "/c:/").replace('\\', "/")
            )
            .as_str(),
        )
        .unwrap()
    }

    fn create(&mut self, fixture_path: &str, text: String) {
        let url = self.fixture_path(fixture_path);

        self.server.notification::<DidOpenTextDocument>(DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: url.clone(),
                language_id: "rust".to_owned(),
                version: 0,
                text: String::new(),
            },
        });

        self.server.notification::<DidChangeTextDocument>(DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier { uri: url, version: 0 },
            content_changes: vec![TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text,
            }],
        });
    }

    fn delete(&mut self, file_idx: usize) {
        self.server.notification::<DidOpenTextDocument>(DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: self.urls[file_idx].clone(),
                language_id: "rust".to_owned(),
                version: 0,
                text: "".to_owned(),
            },
        });

        // See if deleting ratoml file will make the config of interest to return to its default value.
        self.server.notification::<DidSaveTextDocument>(DidSaveTextDocumentParams {
            text_document: TextDocumentIdentifier { uri: self.urls[file_idx].clone() },
            text: Some("".to_owned()),
        });
    }

    fn edit(&mut self, file_idx: usize, text: String) {
        self.server.notification::<DidOpenTextDocument>(DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: self.urls[file_idx].clone(),
                language_id: "rust".to_owned(),
                version: 0,
                text: String::new(),
            },
        });

        self.server.notification::<DidChangeTextDocument>(DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier {
                uri: self.urls[file_idx].clone(),
                version: 0,
            },
            content_changes: vec![TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text,
            }],
        });
    }

    fn query(&self, query: QueryType, source_file_idx: usize) -> bool {
        match query {
            QueryType::AssistEmitMustUse => {
                let res = self.server.send_request::<CodeActionRequest>(CodeActionParams {
                    text_document: TextDocumentIdentifier {
                        uri: self.urls[source_file_idx].clone(),
                    },
                    range: lsp_types::Range {
                        start: Position::new(2, 13),
                        end: Position::new(2, 15),
                    },
                    context: CodeActionContext {
                        diagnostics: vec![],
                        only: None,
                        trigger_kind: None,
                    },
                    work_done_progress_params: WorkDoneProgressParams { work_done_token: None },
                    partial_result_params: lsp_types::PartialResultParams {
                        partial_result_token: None,
                    },
                });

                let res = serde_json::de::from_str::<CodeActionResponse>(res.to_string().as_str())
                    .unwrap();

                // The difference setting the new config key will cause can be seen in the lower layers of this nested response
                // so here are some ugly unwraps and other obscure stuff.
                let ca: CodeAction = res
                    .into_iter()
                    .find_map(|it| {
                        if let CodeActionOrCommand::CodeAction(ca) = it {
                            if ca.title.as_str() == "Generate an `as_` method for this enum variant"
                            {
                                return Some(ca);
                            }
                        }

                        None
                    })
                    .unwrap();

                if let lsp_types::DocumentChanges::Edits(edits) =
                    ca.edit.unwrap().document_changes.unwrap()
                {
                    if let lsp_types::OneOf::Left(l) = &edits[0].edits[0] {
                        return l.new_text.as_str() == RatomlTest::EMIT_MUST_USE_SNIPPET;
                    }
                }
            }
            QueryType::GlobalHover => {
                let res = self.server.send_request::<HoverRequest>(HoverParams {
                    work_done_progress_params: WorkDoneProgressParams { work_done_token: None },
                    text_document_position_params: TextDocumentPositionParams {
                        text_document: TextDocumentIdentifier {
                            uri: self.urls[source_file_idx].clone(),
                        },
                        position: Position::new(7, 18),
                    },
                });
                let res = serde_json::de::from_str::<Hover>(res.to_string().as_str()).unwrap();
                assert!(matches!(res.contents, lsp_types::HoverContents::Markup(_)));
                if let lsp_types::HoverContents::Markup(m) = res.contents {
                    return m.value == RatomlTest::GLOBAL_TRAIT_ASSOC_ITEMS_SNIPPET;
                }
            }
        }

        panic!()
    }
}

static INIT: Once = Once::new();

fn setup() {
    INIT.call_once(|| {
        let trc = rust_analyzer::tracing::Config {
            writer: TestWriter::default(),
            // Deliberately enable all `error` logs if the user has not set RA_LOG, as there is usually
            // useful information in there for debugging.
            filter: std::env::var("RA_LOG").ok().unwrap_or_else(|| "error".to_owned()),
            chalk_filter: std::env::var("CHALK_DEBUG").ok(),
            profile_filter: std::env::var("RA_PROFILE").ok(),
        };

        trc.init().unwrap();
    });
}

// /// Check if we are listening for changes in user's config file ( e.g on Linux `~/.config/rust-analyzer/.rust-analyzer.toml`)
// #[test]
// #[cfg(target_os = "windows")]
// fn listen_to_user_config_scenario_windows() {
//     todo!()
// }

// #[test]
// #[cfg(target_os = "linux")]
// fn listen_to_user_config_scenario_linux() {
//     todo!()
// }

// #[test]
// #[cfg(target_os = "macos")]
// fn listen_to_user_config_scenario_macos() {
//     todo!()
// }

/// Check if made changes have had any effect on
/// the client config.
#[test]
fn ratoml_client_config_basic() {
    let server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
[package]
name = "p1"
version = "0.1.0"
edition = "2021"
"#,
            r#"//- /p1/src/lib.rs
enum Value {
    Number(i32),
    Text(String),
}"#,
        ],
        vec!["p1"],
        Some(json!({
            "assist" : {
                "emitMustUse" : true
            }
        })),
    );

    assert!(server.query(QueryType::AssistEmitMustUse, 1));
}

/// Checks if client config can be modified.
/// FIXME @alibektas : This test is atm not valid.
/// Asking for client config from the client is a 2 way communication
/// which we cannot imitate with the current slow-tests infrastructure.
/// See rust-analyzer::handlers::notifications#197
//     #[test]
//     fn client_config_update() {
//         setup();

//         let server = RatomlTest::new(
//             vec![
//                 r#"
// //- /p1/Cargo.toml
// [package]
// name = "p1"
// version = "0.1.0"
// edition = "2021"
// "#,
//                 r#"
// //- /p1/src/lib.rs
// enum Value {
//     Number(i32),
//     Text(String),
// }"#,
//             ],
//             vec!["p1"],
//             None,
//         );

//         assert!(!server.query(QueryType::AssistEmitMustUse, 1));

//         // a.notification::<DidChangeConfiguration>(DidChangeConfigurationParams {
//         //     settings: json!({
//         //         "assists" : {
//         //             "emitMustUse" : true
//         //         }
//         //     }),
//         // });

//         assert!(server.query(QueryType::AssistEmitMustUse, 1));
//     }

//     #[test]
//     fn ratoml_create_ratoml_basic() {
//         let server = RatomlTest::new(
//             vec![
//                 r#"
// //- /p1/Cargo.toml
// [package]
// name = "p1"
// version = "0.1.0"
// edition = "2021"
// "#,
//                 r#"
// //- /p1/rust-analyzer.toml
// assist.emitMustUse = true
// "#,
//                 r#"
// //- /p1/src/lib.rs
// enum Value {
//     Number(i32),
//     Text(String),
// }
// "#,
//             ],
//             vec!["p1"],
//             None,
//         );

//         assert!(server.query(QueryType::AssistEmitMustUse, 2));
//     }

#[test]
fn ratoml_user_config_detected() {
    let server = RatomlTest::new(
        vec![
            r#"
//- /$$CONFIG_DIR$$/rust-analyzer/rust-analyzer.toml
assist.emitMustUse = true
"#,
            r#"
//- /p1/Cargo.toml
[package]
name = "p1"
version = "0.1.0"
edition = "2021"
"#,
            r#"//- /p1/src/lib.rs
enum Value {
    Number(i32),
    Text(String),
}"#,
        ],
        vec!["p1"],
        None,
    );

    assert!(server.query(QueryType::AssistEmitMustUse, 2));
}

#[test]
fn ratoml_create_user_config() {
    setup();
    let mut server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
[package]
name = "p1"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/src/lib.rs
enum Value {
    Number(i32),
    Text(String),
}"#,
        ],
        vec!["p1"],
        None,
    );

    assert!(!server.query(QueryType::AssistEmitMustUse, 1));
    server.create(
        "//- /$$CONFIG_DIR$$/rust-analyzer/rust-analyzer.toml",
        RatomlTest::EMIT_MUST_USE.to_owned(),
    );
    assert!(server.query(QueryType::AssistEmitMustUse, 1));
}

#[test]
fn ratoml_modify_user_config() {
    let mut server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
[package]
name = "p1"
version = "0.1.0"
edition = "2021""#,
            r#"
//- /p1/src/lib.rs
enum Value {
    Number(i32),
    Text(String),
}"#,
            r#"
//- /$$CONFIG_DIR$$/rust-analyzer/rust-analyzer.toml
assist.emitMustUse = true"#,
        ],
        vec!["p1"],
        None,
    );

    assert!(server.query(QueryType::AssistEmitMustUse, 1));
    server.edit(2, String::new());
    assert!(!server.query(QueryType::AssistEmitMustUse, 1));
}

#[test]
fn ratoml_delete_user_config() {
    let mut server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
[package]
name = "p1"
version = "0.1.0"
edition = "2021""#,
            r#"
//- /p1/src/lib.rs
enum Value {
    Number(i32),
    Text(String),
}"#,
            r#"
//- /$$CONFIG_DIR$$/rust-analyzer/rust-analyzer.toml
assist.emitMustUse = true"#,
        ],
        vec!["p1"],
        None,
    );

    assert!(server.query(QueryType::AssistEmitMustUse, 1));
    server.delete(2);
    assert!(!server.query(QueryType::AssistEmitMustUse, 1));
}
// #[test]
// fn delete_user_config() {
//     todo!()
// }

// #[test]
// fn modify_client_config() {
//     todo!()
// }

#[test]
fn ratoml_inherit_config_from_ws_root() {
    let server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
workspace = { members = ["p2"] }
[package]
name = "p1"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/rust-analyzer.toml
assist.emitMustUse = true
"#,
            r#"
//- /p1/p2/Cargo.toml
[package]
name = "p2"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/p2/src/lib.rs
enum Value {
    Number(i32),
    Text(String),
}"#,
            r#"
//- /p1/src/lib.rs
pub fn add(left: usize, right: usize) -> usize {
    left + right
}
"#,
        ],
        vec!["p1"],
        None,
    );

    assert!(server.query(QueryType::AssistEmitMustUse, 3));
}

#[test]
fn ratoml_modify_ratoml_at_ws_root() {
    let mut server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
workspace = { members = ["p2"] }
[package]
name = "p1"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/rust-analyzer.toml
assist.emitMustUse = false
"#,
            r#"
//- /p1/p2/Cargo.toml
[package]
name = "p2"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/p2/src/lib.rs
enum Value {
    Number(i32),
    Text(String),
}"#,
            r#"
//- /p1/src/lib.rs
pub fn add(left: usize, right: usize) -> usize {
    left + right
}
"#,
        ],
        vec!["p1"],
        None,
    );

    assert!(!server.query(QueryType::AssistEmitMustUse, 3));
    server.edit(1, "assist.emitMustUse = true".to_owned());
    assert!(server.query(QueryType::AssistEmitMustUse, 3));
}

#[test]
fn ratoml_delete_ratoml_at_ws_root() {
    let mut server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
workspace = { members = ["p2"] }
[package]
name = "p1"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/rust-analyzer.toml
assist.emitMustUse = true
"#,
            r#"
//- /p1/p2/Cargo.toml
[package]
name = "p2"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/p2/src/lib.rs
enum Value {
    Number(i32),
    Text(String),
}"#,
            r#"
//- /p1/src/lib.rs
pub fn add(left: usize, right: usize) -> usize {
    left + right
}
"#,
        ],
        vec!["p1"],
        None,
    );

    assert!(server.query(QueryType::AssistEmitMustUse, 3));
    server.delete(1);
    assert!(!server.query(QueryType::AssistEmitMustUse, 3));
}

#[test]
fn ratoml_add_immediate_child_to_ws_root() {
    let mut server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
workspace = { members = ["p2"] }
[package]
name = "p1"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/rust-analyzer.toml
assist.emitMustUse = true
"#,
            r#"
//- /p1/p2/Cargo.toml
[package]
name = "p2"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/p2/src/lib.rs
enum Value {
    Number(i32),
    Text(String),
}"#,
            r#"
//- /p1/src/lib.rs
pub fn add(left: usize, right: usize) -> usize {
    left + right
}
"#,
        ],
        vec!["p1"],
        None,
    );

    assert!(server.query(QueryType::AssistEmitMustUse, 3));
    server.create("//- /p1/p2/rust-analyzer.toml", RatomlTest::EMIT_MUST_NOT_USE.to_owned());
    assert!(!server.query(QueryType::AssistEmitMustUse, 3));
}

#[test]
fn ratoml_rm_ws_root_ratoml_child_has_client_as_parent_now() {
    let mut server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
workspace = { members = ["p2"] }
[package]
name = "p1"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/rust-analyzer.toml
assist.emitMustUse = true
"#,
            r#"
//- /p1/p2/Cargo.toml
[package]
name = "p2"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/p2/src/lib.rs
enum Value {
    Number(i32),
    Text(String),
}"#,
            r#"
//- /p1/src/lib.rs
pub fn add(left: usize, right: usize) -> usize {
    left + right
}
"#,
        ],
        vec!["p1"],
        None,
    );

    assert!(server.query(QueryType::AssistEmitMustUse, 3));
    server.delete(1);
    assert!(!server.query(QueryType::AssistEmitMustUse, 3));
}

#[test]
fn ratoml_crates_both_roots() {
    let server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
workspace = { members = ["p2"] }
[package]
name = "p1"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/rust-analyzer.toml
assist.emitMustUse = true
"#,
            r#"
//- /p1/p2/Cargo.toml
[package]
name = "p2"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/p2/src/lib.rs
enum Value {
    Number(i32),
    Text(String),
}"#,
            r#"
//- /p1/src/lib.rs
enum Value {
    Number(i32),
    Text(String),
}"#,
        ],
        vec!["p1", "p2"],
        None,
    );

    assert!(server.query(QueryType::AssistEmitMustUse, 3));
    assert!(server.query(QueryType::AssistEmitMustUse, 4));
}

#[test]
fn ratoml_multiple_ratoml_in_single_source_root() {
    let server = RatomlTest::new(
        vec![
            r#"
        //- /p1/Cargo.toml
        [package]
        name = "p1"
        version = "0.1.0"
        edition = "2021"
        "#,
            r#"
        //- /p1/rust-analyzer.toml
        assist.emitMustUse = true
        "#,
            r#"
        //- /p1/src/rust-analyzer.toml
        assist.emitMustUse = false
        "#,
            r#"
        //- /p1/src/lib.rs
        enum Value {
            Number(i32),
            Text(String),
        }
        "#,
        ],
        vec!["p1"],
        None,
    );

    assert!(server.query(QueryType::AssistEmitMustUse, 3));

    let server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
[package]
name = "p1"
version = "0.1.0"
edition = "2021"
"#,
            r#"
//- /p1/src/rust-analyzer.toml
assist.emitMustUse = false
"#,
            r#"
//- /p1/rust-analyzer.toml
assist.emitMustUse = true
"#,
            r#"
//- /p1/src/lib.rs
enum Value {
    Number(i32),
    Text(String),
}
"#,
        ],
        vec!["p1"],
        None,
    );

    assert!(server.query(QueryType::AssistEmitMustUse, 3));
}

/// If a root is non-local, so we cannot find what its parent is
/// in our `config.local_root_parent_map`. So if any config should
/// apply, it must be looked for starting from the client level.
/// FIXME @alibektas : "locality" is according to ra that, which is simply in the file system.
/// This doesn't really help us with what we want to achieve here.
//     #[test]
//     fn ratoml_non_local_crates_start_inheriting_from_client() {
//         let server = RatomlTest::new(
//             vec![
//                 r#"
// //- /p1/Cargo.toml
// [package]
// name = "p1"
// version = "0.1.0"
// edition = "2021"

// [dependencies]
// p2 = { path = "../p2" }
// #,
//                 r#"
// //- /p1/src/lib.rs
// enum Value {
//     Number(i32),
//     Text(String),
// }

// use p2;

// pub fn add(left: usize, right: usize) -> usize {
//     p2::add(left, right)
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }"#,
//                 r#"
// //- /p2/Cargo.toml
// [package]
// name = "p2"
// version = "0.1.0"
// edition = "2021"

// # See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

// [dependencies]
// "#,
//                 r#"
// //- /p2/rust-analyzer.toml
// # DEF
// assist.emitMustUse = true
// "#,
//                 r#"
// //- /p2/src/lib.rs
// enum Value {
//     Number(i32),
//     Text(String),
// }"#,
//             ],
//             vec!["p1", "p2"],
//             None,
//         );

//         assert!(!server.query(QueryType::AssistEmitMustUse, 5));
//     }

/// Having a ratoml file at the root of a project enables
/// configuring global level configurations as well.
#[test]
fn ratoml_in_root_is_global() {
    let server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
[package]
name = "p1"
version = "0.1.0"
edition = "2021"
        "#,
            r#"
//- /rust-analyzer.toml
hover.show.traitAssocItems = 4
        "#,
            r#"
//- /p1/src/lib.rs
trait RandomTrait {
    type B;
    fn abc() -> i32;
    fn def() -> i64;
}

fn main() {
    let a = RandomTrait;
}"#,
        ],
        vec![],
        None,
    );

    server.query(QueryType::GlobalHover, 2);
}

#[test]
fn ratoml_root_is_updateable() {
    let mut server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
[package]
name = "p1"
version = "0.1.0"
edition = "2021"
        "#,
            r#"
//- /rust-analyzer.toml
hover.show.traitAssocItems = 4
        "#,
            r#"
//- /p1/src/lib.rs
trait RandomTrait {
    type B;
    fn abc() -> i32;
    fn def() -> i64;
}

fn main() {
    let a = RandomTrait;
}"#,
        ],
        vec![],
        None,
    );

    assert!(server.query(QueryType::GlobalHover, 2));
    server.edit(1, RatomlTest::GLOBAL_TRAIT_ASSOC_ITEMS_ZERO.to_owned());
    assert!(!server.query(QueryType::GlobalHover, 2));
}

#[test]
fn ratoml_root_is_deletable() {
    let mut server = RatomlTest::new(
        vec![
            r#"
//- /p1/Cargo.toml
[package]
name = "p1"
version = "0.1.0"
edition = "2021"
        "#,
            r#"
//- /rust-analyzer.toml
hover.show.traitAssocItems = 4
        "#,
            r#"
//- /p1/src/lib.rs
trait RandomTrait {
    type B;
    fn abc() -> i32;
    fn def() -> i64;
}

fn main() {
    let a = RandomTrait;
}"#,
        ],
        vec![],
        None,
    );

    assert!(server.query(QueryType::GlobalHover, 2));
    server.delete(1);
    assert!(!server.query(QueryType::GlobalHover, 2));
}
