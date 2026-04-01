use crate::support::{Project, Server};
use crate::testdir::TestDir;
use lsp_types::{
    DidChangeTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams,
    TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem, Url,
    VersionedTextDocumentIdentifier,
    notification::{DidChangeTextDocument, DidOpenTextDocument, DidSaveTextDocument},
};
use paths::Utf8PathBuf;

use rust_analyzer::config::Config;
use rust_analyzer::lsp::ext::{
    InternalTestingFetchConfig, InternalTestingFetchConfigOption, InternalTestingFetchConfigParams,
    InternalTestingFetchConfigResponse,
};
use serde_json::json;
use test_utils::skip_slow_tests;

struct RatomlTest {
    urls: Vec<Url>,
    server: Server,
    tmp_path: Utf8PathBuf,
}

impl RatomlTest {
    const EMIT_MUST_USE: &'static str = r#"assist.emitMustUse = true"#;
    const EMIT_MUST_NOT_USE: &'static str = r#"assist.emitMustUse = false"#;

    fn new(
        fixtures: Vec<&str>,
        roots: Vec<&str>,
        client_config: Option<serde_json::Value>,
    ) -> Self {
        let tmp_dir = TestDir::new();
        let tmp_path = tmp_dir.path().to_owned();

        let full_fixture = fixtures.join("\n");

        let mut project = Project::with_fixture(&full_fixture).tmp_dir(tmp_dir);

        for root in roots {
            project = project.root(root);
        }

        if let Some(client_config) = client_config {
            project = project.with_config(client_config);
        }

        let server = project.server_with_lock(true).wait_until_workspace_is_loaded();

        let mut case = Self { urls: vec![], server, tmp_path };
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
                path = Config::user_config_dir_path().unwrap().into();
            } else {
                path = path.join(first);
            }
        }
        for piece in spl {
            path = path.join(piece);
        }

        Url::parse(
            format!("file://{}", path.into_string().replace("C:\\", "/c:/").replace('\\', "/"))
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

    fn query(
        &self,
        query: InternalTestingFetchConfigOption,
        source_file_idx: usize,
        expected: InternalTestingFetchConfigResponse,
    ) {
        let res = self.server.send_request::<InternalTestingFetchConfig>(
            InternalTestingFetchConfigParams {
                text_document: Some(TextDocumentIdentifier {
                    uri: self.urls[source_file_idx].clone(),
                }),
                config: query,
            },
        );
        assert_eq!(
            serde_json::from_value::<InternalTestingFetchConfigResponse>(res).unwrap(),
            expected
        )
    }
}

/// Check if made changes have had any effect on
/// the client config.
#[test]
fn ratoml_client_config_basic() {
    if skip_slow_tests() {
        return;
    }

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

    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        1,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(true),
    );
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
    if skip_slow_tests() {
        return;
    }

    let server = RatomlTest::new(
        vec![
            r#"
//- /$$CONFIG_DIR$$/rust-analyzer.toml
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

    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        2,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(true),
    );
}

#[test]
fn ratoml_create_user_config() {
    if skip_slow_tests() {
        return;
    }

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

    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        1,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(false),
    );
    server.create("//- /$$CONFIG_DIR$$/rust-analyzer.toml", RatomlTest::EMIT_MUST_USE.to_owned());
    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        1,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(true),
    );
}

#[test]
fn ratoml_modify_user_config() {
    if skip_slow_tests() {
        return;
    }

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
//- /$$CONFIG_DIR$$/rust-analyzer.toml
assist.emitMustUse = true"#,
        ],
        vec!["p1"],
        None,
    );

    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        1,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(true),
    );
    server.edit(2, String::new());
    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        1,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(false),
    );
}

#[test]
fn ratoml_delete_user_config() {
    if skip_slow_tests() {
        return;
    }

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
//- /$$CONFIG_DIR$$/rust-analyzer.toml
assist.emitMustUse = true"#,
        ],
        vec!["p1"],
        None,
    );

    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        1,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(true),
    );
    server.delete(2);
    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        1,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(false),
    );
}

#[test]
#[ignore = "flaky test that tends to hang"]
fn ratoml_inherit_config_from_ws_root() {
    if skip_slow_tests() {
        return;
    }

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

    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        3,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(true),
    );
}

#[test]
fn ratoml_modify_ratoml_at_ws_root() {
    if skip_slow_tests() {
        return;
    }

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

    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        3,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(false),
    );
    server.edit(1, "assist.emitMustUse = true".to_owned());
    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        3,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(true),
    );
}

#[test]
fn ratoml_delete_ratoml_at_ws_root() {
    if skip_slow_tests() {
        return;
    }

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

    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        3,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(true),
    );
    server.delete(1);
    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        3,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(false),
    );
}

#[test]
fn ratoml_add_immediate_child_to_ws_root() {
    if skip_slow_tests() {
        return;
    }

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

    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        3,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(true),
    );
    server.create("//- /p1/p2/rust-analyzer.toml", RatomlTest::EMIT_MUST_NOT_USE.to_owned());
    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        3,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(false),
    );
}

#[test]
#[ignore = "Root ratomls are not being looked for on startup. Fix this."]
fn ratoml_rm_ws_root_ratoml_child_has_client_as_parent_now() {
    if skip_slow_tests() {
        return;
    }

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

    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        3,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(true),
    );
    server.delete(1);
    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        3,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(false),
    );
}

#[test]
fn ratoml_crates_both_roots() {
    if skip_slow_tests() {
        return;
    }

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

    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        3,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(true),
    );
    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        4,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(true),
    );
}

#[test]
fn ratoml_multiple_ratoml_in_single_source_root() {
    if skip_slow_tests() {
        return;
    }

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

    server.query(
        InternalTestingFetchConfigOption::AssistEmitMustUse,
        3,
        InternalTestingFetchConfigResponse::AssistEmitMustUse(true),
    );
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
// "#,
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

#[test]
fn ratoml_in_root_is_workspace() {
    if skip_slow_tests() {
        return;
    }

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
check.workspace = false
        "#,
            r#"
//- /p1/src/lib.rs
fn main() {
    todo!()
}"#,
        ],
        vec!["p1"],
        None,
    );

    server.query(
        InternalTestingFetchConfigOption::CheckWorkspace,
        2,
        InternalTestingFetchConfigResponse::CheckWorkspace(false),
    )
}

#[test]
fn ratoml_root_is_updateable() {
    if skip_slow_tests() {
        return;
    }

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
//- /p1/rust-analyzer.toml
check.workspace = false
    "#,
            r#"
//- /p1/src/lib.rs
fn main() {
   todo!()
}"#,
        ],
        vec!["p1"],
        None,
    );

    server.query(
        InternalTestingFetchConfigOption::CheckWorkspace,
        2,
        InternalTestingFetchConfigResponse::CheckWorkspace(false),
    );
    server.edit(1, "check.workspace = true".to_owned());
    server.query(
        InternalTestingFetchConfigOption::CheckWorkspace,
        2,
        InternalTestingFetchConfigResponse::CheckWorkspace(true),
    );
}

#[test]
fn ratoml_root_is_deletable() {
    if skip_slow_tests() {
        return;
    }

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
//- /p1/rust-analyzer.toml
check.workspace = false
       "#,
            r#"
//- /p1/src/lib.rs
fn main() {
    todo!()
}"#,
        ],
        vec!["p1"],
        None,
    );

    server.query(
        InternalTestingFetchConfigOption::CheckWorkspace,
        2,
        InternalTestingFetchConfigResponse::CheckWorkspace(false),
    );
    server.delete(1);
    server.query(
        InternalTestingFetchConfigOption::CheckWorkspace,
        2,
        InternalTestingFetchConfigResponse::CheckWorkspace(true),
    );
}
