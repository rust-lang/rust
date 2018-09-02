#[macro_use]
extern crate crossbeam_channel;
extern crate tempdir;
extern crate languageserver_types;
extern crate serde;
extern crate serde_json;
extern crate gen_lsp_server;
extern crate flexi_logger;
extern crate m;

mod support;

use m::req::{Runnables, RunnablesParams, DidReloadWorkspace};

use support::project;

const LOG: &'static str = "WARN";

#[test]
fn test_runnables() {
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
fn test_project_model() {
    let server = project(r#"
//- Cargo.toml
[package]
name = "foo"
version = "0.0.0"

//- src/lib.rs
pub fn foo() {}
"#);
    server.notification::<DidReloadWorkspace>(r#"[
  {
    "packages": [
      {
        "manifest": "$PROJECT_ROOT$/Cargo.toml",
        "name": "foo",
        "targets": [ 0 ]
      }
    ],
    "targets": [
      { "kind": "Lib", "name": "foo", "pkg": 0, "root": "$PROJECT_ROOT$/src/lib.rs" }
    ],
    "ws_members": [ 0 ]
  }
]"#
    );
}
