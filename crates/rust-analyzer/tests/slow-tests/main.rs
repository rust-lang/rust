//! The most high-level integrated tests for rust-analyzer.
//!
//! This tests run a full LSP event loop, spawn cargo and process stdlib from
//! sysroot. For this reason, the tests here are very slow, and should be
//! avoided unless absolutely necessary.
//!
//! In particular, it's fine *not* to test that client & server agree on
//! specific JSON shapes here -- there's little value in such tests, as we can't
//! be sure without a real client anyway.

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

#[cfg(not(feature = "in-rust-tree"))]
mod sourcegen;
mod support;
mod testdir;
mod tidy;

use std::{collections::HashMap, path::PathBuf, time::Instant};

use lsp_types::{
    notification::DidOpenTextDocument,
    request::{
        CodeActionRequest, Completion, Formatting, GotoTypeDefinition, HoverRequest,
        WillRenameFiles, WorkspaceSymbolRequest,
    },
    CodeActionContext, CodeActionParams, CompletionParams, DidOpenTextDocumentParams,
    DocumentFormattingParams, FileRename, FormattingOptions, GotoDefinitionParams, HoverParams,
    PartialResultParams, Position, Range, RenameFilesParams, TextDocumentItem,
    TextDocumentPositionParams, WorkDoneProgressParams,
};
use rust_analyzer::lsp_ext::{OnEnter, Runnables, RunnablesParams};
use serde_json::json;
use test_utils::skip_slow_tests;

use crate::{
    support::{project, Project},
    testdir::TestDir,
};

const PROFILE: &str = "";
// const PROFILE: &'static str = "*@3>100";

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
              "executableArgs": ["test_eggs", "--exact", "--nocapture"],
              "cargoExtraArgs": [],
              "overrideCargo": null,
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
                "--nocapture"
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
fn diagnostics_dont_block_typing() {
    if skip_slow_tests() {
        return;
    }

    let librs: String = (0..10).map(|i| format!("mod m{i};")).collect();
    let libs: String = (0..10).map(|i| format!("//- /src/m{i}.rs\nfn foo() {{}}\n\n")).collect();
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
                language_id: "rust".to_string(),
                version: 0,
                text: "/// Docs\nfn foo() {}".to_string(),
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

#[test]
fn out_dirs_check() {
    if skip_slow_tests() {
        return;
    }

    let server = Project::with_fixture(
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
    )
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
#[cfg(feature = "sysroot-abi")]
fn resolve_proc_macro() {
    use expect_test::expect;
    if skip_slow_tests() {
        return;
    }

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
            "server": PathBuf::from(env!("CARGO_BIN_EXE_rust-analyzer")),
        }
    }))
    .root("foo")
    .root("bar")
    .server()
    .wait_until_workspace_is_loaded();

    let res = server.send_request::<HoverRequest>(HoverParams {
        text_document_position_params: TextDocumentPositionParams::new(
            server.doc_id("foo/src/main.rs"),
            Position::new(11, 9),
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
    let tmp_dir_str = tmp_dir_path.to_str().unwrap();
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

//- /src/from_mod/mod.rs

//- /src/to_mod/foo.rs

"#;
    let server =
        Project::with_fixture(code).tmp_dir(tmp_dir).server().wait_until_workspace_is_loaded();

    //rename same level file
    server.request::<WillRenameFiles>(
        RenameFilesParams {
            files: vec![FileRename {
                old_uri: base_path.join("src/old_file.rs").to_str().unwrap().to_string(),
                new_uri: base_path.join("src/new_file.rs").to_str().unwrap().to_string(),
            }],
        },
        json!({
          "documentChanges": [
            {
              "textDocument": {
                "uri": format!("file://{}", tmp_dir_path.join("src").join("lib.rs").to_str().unwrap().to_string().replace("C:\\", "/c:/").replace('\\', "/")),
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
                old_uri: base_path.join("src/from_mod/mod.rs").to_str().unwrap().to_string(),
                new_uri: base_path.join("src/from_mod/foo.rs").to_str().unwrap().to_string(),
            }],
        },
        json!(null),
    );

    //rename file from foo.rs to mod.rs
    server.request::<WillRenameFiles>(
        RenameFilesParams {
            files: vec![FileRename {
                old_uri: base_path.join("src/to_mod/foo.rs").to_str().unwrap().to_string(),
                new_uri: base_path.join("src/to_mod/mod.rs").to_str().unwrap().to_string(),
            }],
        },
        json!(null),
    );

    //rename same level file
    server.request::<WillRenameFiles>(
        RenameFilesParams {
            files: vec![FileRename {
                old_uri: base_path.join("src/old_folder").to_str().unwrap().to_string(),
                new_uri: base_path.join("src/new_folder").to_str().unwrap().to_string(),
            }],
        },
        json!({
          "documentChanges": [
            {
              "textDocument": {
                "uri": format!("file://{}", tmp_dir_path.join("src").join("lib.rs").to_str().unwrap().to_string().replace("C:\\", "/c:/").replace('\\', "/")),
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
