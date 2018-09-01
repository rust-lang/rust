extern crate tempdir;
extern crate crossbeam_channel;
extern crate languageserver_types;
extern crate serde;
extern crate serde_json;
extern crate gen_lsp_server;
extern crate flexi_logger;
extern crate m;

mod support;

use m::req::{Runnables, RunnablesParams};

use support::project;

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
