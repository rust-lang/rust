use crate::support::{Project, Server};
use crate::testdir::TestDir;
use lsp_types::{
    notification::{DidChangeTextDocument, DidOpenTextDocument, DidSaveTextDocument},
    DidChangeTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams,
    TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem, Url,
    VersionedTextDocumentIdentifier,
};
use paths::Utf8PathBuf;

use rust_analyzer::lsp::ext::{InternalTestingFetchConfig, InternalTestingFetchConfigParams};
use serde_json::json;

enum QueryType {
    Local,
    /// A query whose config key is a part of the global configs, so that
    /// testing for changes to this config means testing if global changes
    /// take affect.
    Global,
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

    const GLOBAL_TRAIT_ASSOC_ITEMS_ZERO: &'static str = r#"hover.show.traitAssocItems = 0"#;

    fn new(
        fixtures: Vec<&str>,
        roots: Vec<&str>,
        client_config: Option<serde_json::Value>,
    ) -> Self {
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
        let config = match query {
            QueryType::Local => "local".to_owned(),
            QueryType::Global => "global".to_owned(),
        };
        let res = self.server.send_request::<InternalTestingFetchConfig>(
            InternalTestingFetchConfigParams {
                text_document: Some(TextDocumentIdentifier {
                    uri: self.urls[source_file_idx].clone(),
                }),
                config,
            },
        );
        res.as_bool().unwrap()
    }
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

    assert!(server.query(QueryType::Local, 1));
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
#[ignore = "the user config is currently not being watched on startup, fix this"]
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

    assert!(server.query(QueryType::Local, 2));
}

#[test]
#[ignore = "the user config is currently not being watched on startup, fix this"]
fn ratoml_create_user_config() {
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

    assert!(!server.query(QueryType::Local, 1));
    server.create(
        "//- /$$CONFIG_DIR$$/rust-analyzer/rust-analyzer.toml",
        RatomlTest::EMIT_MUST_USE.to_owned(),
    );
    assert!(server.query(QueryType::Local, 1));
}

#[test]
#[ignore = "the user config is currently not being watched on startup, fix this"]
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

    assert!(server.query(QueryType::Local, 1));
    server.edit(2, String::new());
    assert!(!server.query(QueryType::Local, 1));
}

#[test]
#[ignore = "the user config is currently not being watched on startup, fix this"]
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

    assert!(server.query(QueryType::Local, 1));
    server.delete(2);
    assert!(!server.query(QueryType::Local, 1));
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

    assert!(server.query(QueryType::Local, 3));
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

    assert!(!server.query(QueryType::Local, 3));
    server.edit(1, "assist.emitMustUse = true".to_owned());
    assert!(server.query(QueryType::Local, 3));
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

    assert!(server.query(QueryType::Local, 3));
    server.delete(1);
    assert!(!server.query(QueryType::Local, 3));
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

    assert!(server.query(QueryType::Local, 3));
    server.create("//- /p1/p2/rust-analyzer.toml", RatomlTest::EMIT_MUST_NOT_USE.to_owned());
    assert!(!server.query(QueryType::Local, 3));
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

    assert!(server.query(QueryType::Local, 3));
    server.delete(1);
    assert!(!server.query(QueryType::Local, 3));
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

    assert!(server.query(QueryType::Local, 3));
    assert!(server.query(QueryType::Local, 4));
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

    assert!(server.query(QueryType::Local, 3));

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

    assert!(server.query(QueryType::Local, 3));
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

    server.query(QueryType::Global, 2);
}

#[allow(unused)]
// #[test]
// FIXME: Re-enable this test when we have a global config we can check again
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

    assert!(server.query(QueryType::Global, 2));
    server.edit(1, RatomlTest::GLOBAL_TRAIT_ASSOC_ITEMS_ZERO.to_owned());
    assert!(!server.query(QueryType::Global, 2));
}

#[allow(unused)]
// #[test]
// FIXME: Re-enable this test when we have a global config we can check again
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

    assert!(server.query(QueryType::Global, 2));
    server.delete(1);
    assert!(!server.query(QueryType::Global, 2));
}
