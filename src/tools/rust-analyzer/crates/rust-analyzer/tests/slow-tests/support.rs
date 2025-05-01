use std::{
    cell::{Cell, RefCell},
    env, fs,
    sync::Once,
    time::Duration,
};

use crossbeam_channel::{Receiver, after, select};
use itertools::Itertools;
use lsp_server::{Connection, Message, Notification, Request};
use lsp_types::{TextDocumentIdentifier, Url, notification::Exit, request::Shutdown};
use parking_lot::{Mutex, MutexGuard};
use paths::{Utf8Path, Utf8PathBuf};
use rust_analyzer::{
    cli::flags,
    config::{Config, ConfigChange, ConfigErrors},
    lsp, main_loop,
};
use serde::Serialize;
use serde_json::{Value, json, to_string_pretty};
use test_utils::FixtureWithProjectMeta;
use tracing_subscriber::fmt::TestWriter;
use vfs::AbsPathBuf;

use crate::testdir::TestDir;

pub(crate) struct Project<'a> {
    fixture: &'a str,
    tmp_dir: Option<TestDir>,
    roots: Vec<Utf8PathBuf>,
    config: serde_json::Value,
    root_dir_contains_symlink: bool,
}

impl Project<'_> {
    pub(crate) fn with_fixture(fixture: &str) -> Project<'_> {
        Project {
            fixture,
            tmp_dir: None,
            roots: vec![],
            config: serde_json::json!({
                "cargo": {
                    // Loading standard library is costly, let's ignore it by default
                    "sysroot": null,
                    // Can't use test binary as rustc wrapper.
                    "buildScripts": {
                        "useRustcWrapper": false,
                        "enable": false,
                    },
                },
                "procMacro": {
                    "enable": false,
                }
            }),
            root_dir_contains_symlink: false,
        }
    }

    pub(crate) fn tmp_dir(mut self, tmp_dir: TestDir) -> Self {
        self.tmp_dir = Some(tmp_dir);
        self
    }

    pub(crate) fn root(mut self, path: &str) -> Self {
        self.roots.push(path.into());
        self
    }

    pub(crate) fn with_root_dir_contains_symlink(mut self) -> Self {
        self.root_dir_contains_symlink = true;
        self
    }

    pub(crate) fn with_config(mut self, config: serde_json::Value) -> Self {
        fn merge(dst: &mut serde_json::Value, src: serde_json::Value) {
            match (dst, src) {
                (Value::Object(dst), Value::Object(src)) => {
                    for (k, v) in src {
                        merge(dst.entry(k).or_insert(v.clone()), v)
                    }
                }
                (dst, src) => *dst = src,
            }
        }
        merge(&mut self.config, config);
        self
    }

    pub(crate) fn run_lsif(self) -> String {
        let tmp_dir = self.tmp_dir.unwrap_or_else(|| {
            if self.root_dir_contains_symlink { TestDir::new_symlink() } else { TestDir::new() }
        });

        let FixtureWithProjectMeta {
            fixture,
            mini_core,
            proc_macro_names,
            toolchain,
            target_data_layout: _,
        } = FixtureWithProjectMeta::parse(self.fixture);
        assert!(proc_macro_names.is_empty());
        assert!(mini_core.is_none());
        assert!(toolchain.is_none());

        for entry in fixture {
            let path = tmp_dir.path().join(&entry.path['/'.len_utf8()..]);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path.as_path(), entry.text.as_bytes()).unwrap();
        }

        let tmp_dir_path = AbsPathBuf::assert(tmp_dir.path().to_path_buf());
        let mut buf = Vec::new();
        flags::Lsif::run(
            flags::Lsif {
                path: tmp_dir_path.join(self.roots.iter().exactly_one().unwrap()).into(),
                exclude_vendored_libraries: false,
            },
            &mut buf,
            None,
        )
        .unwrap();
        String::from_utf8(buf).unwrap()
    }

    pub(crate) fn server(self) -> Server {
        Project::server_with_lock(self, false)
    }

    /// `prelock` : Forcefully acquire a lock that will maintain the path to the config dir throughout the whole test.
    ///
    /// When testing we set the user config dir by setting an envvar `__TEST_RA_USER_CONFIG_DIR`.
    /// This value must be maintained until the end of a test case. When tests run in parallel
    /// this value may change thus making the tests flaky. As such, we use a `MutexGuard` that locks
    /// the process until `Server` is dropped. To optimize parallelization we use a lock only when it is
    /// needed, that is when a test uses config directory to do stuff. Our naive approach is to use a lock
    /// if there is a path to config dir in the test fixture. However, in certain cases we create a
    /// file in the config dir after server is run, something where our naive approach comes short.
    /// Using a `prelock` allows us to force a lock when we know we need it.
    pub(crate) fn server_with_lock(self, config_lock: bool) -> Server {
        static CONFIG_DIR_LOCK: Mutex<()> = Mutex::new(());

        let config_dir_guard = if config_lock {
            Some({
                let guard = CONFIG_DIR_LOCK.lock();
                let test_dir = TestDir::new();
                let value = test_dir.path().to_owned();
                // SAFETY: This is safe because this is single-threaded.
                unsafe {
                    env::set_var("__TEST_RA_USER_CONFIG_DIR", &value);
                }
                (guard, test_dir)
            })
        } else {
            None
        };

        let tmp_dir = self.tmp_dir.unwrap_or_else(|| {
            if self.root_dir_contains_symlink { TestDir::new_symlink() } else { TestDir::new() }
        });

        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = rust_analyzer::tracing::Config {
                writer: TestWriter::default(),
                // Deliberately enable all `error` logs if the user has not set RA_LOG, as there is usually
                // useful information in there for debugging.
                filter: std::env::var("RA_LOG").ok().unwrap_or_else(|| "error".to_owned()),
                chalk_filter: std::env::var("CHALK_DEBUG").ok(),
                profile_filter: std::env::var("RA_PROFILE").ok(),
                json_profile_filter: std::env::var("RA_PROFILE_JSON").ok(),
            };
        });

        let FixtureWithProjectMeta {
            fixture,
            mini_core,
            proc_macro_names,
            toolchain,
            target_data_layout: _,
        } = FixtureWithProjectMeta::parse(self.fixture);
        assert!(proc_macro_names.is_empty());
        assert!(mini_core.is_none());
        assert!(toolchain.is_none());

        for entry in fixture {
            if let Some(pth) = entry.path.strip_prefix("/$$CONFIG_DIR$$") {
                let path = Config::user_config_dir_path().unwrap().join(&pth['/'.len_utf8()..]);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(path.as_path(), entry.text.as_bytes()).unwrap();
            } else {
                let path = tmp_dir.path().join(&entry.path['/'.len_utf8()..]);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(path.as_path(), entry.text.as_bytes()).unwrap();
            }
        }

        let tmp_dir_path = AbsPathBuf::assert(tmp_dir.path().to_path_buf());
        let mut roots =
            self.roots.into_iter().map(|root| tmp_dir_path.join(root)).collect::<Vec<_>>();
        if roots.is_empty() {
            roots.push(tmp_dir_path.clone());
        }

        let mut config = Config::new(
            tmp_dir_path,
            lsp_types::ClientCapabilities {
                workspace: Some(lsp_types::WorkspaceClientCapabilities {
                    did_change_watched_files: Some(
                        lsp_types::DidChangeWatchedFilesClientCapabilities {
                            dynamic_registration: Some(true),
                            relative_pattern_support: None,
                        },
                    ),
                    workspace_edit: Some(lsp_types::WorkspaceEditClientCapabilities {
                        resource_operations: Some(vec![
                            lsp_types::ResourceOperationKind::Create,
                            lsp_types::ResourceOperationKind::Delete,
                            lsp_types::ResourceOperationKind::Rename,
                        ]),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                text_document: Some(lsp_types::TextDocumentClientCapabilities {
                    definition: Some(lsp_types::GotoCapability {
                        link_support: Some(true),
                        ..Default::default()
                    }),
                    code_action: Some(lsp_types::CodeActionClientCapabilities {
                        code_action_literal_support: Some(
                            lsp_types::CodeActionLiteralSupport::default(),
                        ),
                        ..Default::default()
                    }),
                    hover: Some(lsp_types::HoverClientCapabilities {
                        content_format: Some(vec![lsp_types::MarkupKind::Markdown]),
                        ..Default::default()
                    }),
                    inlay_hint: Some(lsp_types::InlayHintClientCapabilities {
                        resolve_support: Some(lsp_types::InlayHintResolveClientCapabilities {
                            properties: vec![
                                "textEdits".to_owned(),
                                "tooltip".to_owned(),
                                "label.tooltip".to_owned(),
                                "label.location".to_owned(),
                                "label.command".to_owned(),
                            ],
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                window: Some(lsp_types::WindowClientCapabilities {
                    work_done_progress: Some(false),
                    ..Default::default()
                }),
                experimental: Some(json!({
                    "serverStatusNotification": true,
                })),
                ..Default::default()
            },
            roots,
            None,
        );
        let mut change = ConfigChange::default();

        change.change_client_config(self.config);

        let error_sink: ConfigErrors;
        (config, error_sink, _) = config.apply_change(change);
        assert!(error_sink.is_empty(), "{error_sink:?}");

        config.rediscover_workspaces();

        Server::new(config_dir_guard, tmp_dir.keep(), config)
    }
}

pub(crate) fn project(fixture: &str) -> Server {
    Project::with_fixture(fixture).server()
}

pub(crate) struct Server {
    req_id: Cell<i32>,
    messages: RefCell<Vec<Message>>,
    _thread: stdx::thread::JoinHandle,
    client: Connection,
    /// XXX: remove the tempdir last
    dir: TestDir,
    _config_dir_guard: Option<(MutexGuard<'static, ()>, TestDir)>,
}

impl Server {
    fn new(
        config_dir_guard: Option<(MutexGuard<'static, ()>, TestDir)>,
        dir: TestDir,
        config: Config,
    ) -> Server {
        let (connection, client) = Connection::memory();

        let _thread = stdx::thread::Builder::new(stdx::thread::ThreadIntent::Worker, "test server")
            .spawn(move || main_loop(config, connection).unwrap())
            .expect("failed to spawn a thread");

        Server {
            req_id: Cell::new(1),
            dir,
            messages: Default::default(),
            client,
            _thread,
            _config_dir_guard: config_dir_guard,
        }
    }

    pub(crate) fn doc_id(&self, rel_path: &str) -> TextDocumentIdentifier {
        let path = self.dir.path().join(rel_path);
        TextDocumentIdentifier { uri: Url::from_file_path(path).unwrap() }
    }

    pub(crate) fn notification<N>(&self, params: N::Params)
    where
        N: lsp_types::notification::Notification,
        N::Params: Serialize,
    {
        let r = Notification::new(N::METHOD.to_owned(), params);
        self.send_notification(r)
    }

    #[track_caller]
    pub(crate) fn request<R>(&self, params: R::Params, expected_resp: Value)
    where
        R: lsp_types::request::Request,
        R::Params: Serialize,
    {
        let actual = self.send_request::<R>(params);
        if let Some((expected_part, actual_part)) = find_mismatch(&expected_resp, &actual) {
            panic!(
                "JSON mismatch\nExpected:\n{}\nWas:\n{}\nExpected part:\n{}\nActual part:\n{}\n",
                to_string_pretty(&expected_resp).unwrap(),
                to_string_pretty(&actual).unwrap(),
                to_string_pretty(expected_part).unwrap(),
                to_string_pretty(actual_part).unwrap(),
            );
        }
    }

    #[track_caller]
    pub(crate) fn send_request<R>(&self, params: R::Params) -> Value
    where
        R: lsp_types::request::Request,
        R::Params: Serialize,
    {
        let id = self.req_id.get();
        self.req_id.set(id.wrapping_add(1));

        let r = Request::new(id.into(), R::METHOD.to_owned(), params);
        self.send_request_(r)
    }
    #[track_caller]
    fn send_request_(&self, r: Request) -> Value {
        let id = r.id.clone();
        self.client.sender.send(r.clone().into()).unwrap();
        while let Some(msg) = self.recv().unwrap_or_else(|Timeout| panic!("timeout: {r:?}")) {
            match msg {
                Message::Request(req) => {
                    if req.method == "client/registerCapability" {
                        let params = req.params.to_string();
                        if ["workspace/didChangeWatchedFiles", "textDocument/didSave"]
                            .into_iter()
                            .any(|it| params.contains(it))
                        {
                            continue;
                        }
                    }
                    panic!("unexpected request: {req:?}")
                }
                Message::Notification(_) => (),
                Message::Response(res) => {
                    assert_eq!(res.id, id);
                    if let Some(err) = res.error {
                        panic!("error response: {err:#?}");
                    }
                    return res.result.unwrap();
                }
            }
        }
        panic!("no response for {r:?}");
    }
    pub(crate) fn wait_until_workspace_is_loaded(self) -> Server {
        self.wait_for_message_cond(1, &|msg: &Message| match msg {
            Message::Notification(n) if n.method == "experimental/serverStatus" => {
                let status = n
                    .clone()
                    .extract::<lsp::ext::ServerStatusParams>("experimental/serverStatus")
                    .unwrap();
                if status.health != lsp::ext::Health::Ok {
                    panic!("server errored/warned while loading workspace: {:?}", status.message);
                }
                status.quiescent
            }
            _ => false,
        })
        .unwrap_or_else(|Timeout| panic!("timeout while waiting for ws to load"));
        self
    }
    fn wait_for_message_cond(
        &self,
        n: usize,
        cond: &dyn Fn(&Message) -> bool,
    ) -> Result<(), Timeout> {
        let mut total = 0;
        for msg in self.messages.borrow().iter() {
            if cond(msg) {
                total += 1
            }
        }
        while total < n {
            let msg = self.recv()?.expect("no response");
            if cond(&msg) {
                total += 1;
            }
        }
        Ok(())
    }
    fn recv(&self) -> Result<Option<Message>, Timeout> {
        let msg = recv_timeout(&self.client.receiver)?;
        let msg = msg.inspect(|msg| {
            self.messages.borrow_mut().push(msg.clone());
        });
        Ok(msg)
    }
    fn send_notification(&self, not: Notification) {
        self.client.sender.send(Message::Notification(not)).unwrap();
    }

    pub(crate) fn path(&self) -> &Utf8Path {
        self.dir.path()
    }

    pub(crate) fn write_file_and_save(&self, path: &str, text: String) {
        fs::write(self.dir.path().join(path), &text).unwrap();
        self.notification::<lsp_types::notification::DidSaveTextDocument>(
            lsp_types::DidSaveTextDocumentParams {
                text_document: self.doc_id(path),
                text: Some(text),
            },
        )
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.request::<Shutdown>((), Value::Null);
        self.notification::<Exit>(());
    }
}

struct Timeout;

fn recv_timeout(receiver: &Receiver<Message>) -> Result<Option<Message>, Timeout> {
    let timeout =
        if cfg!(target_os = "macos") { Duration::from_secs(300) } else { Duration::from_secs(120) };
    select! {
        recv(receiver) -> msg => Ok(msg.ok()),
        recv(after(timeout)) -> _ => Err(Timeout),
    }
}

// Comparison functionality borrowed from cargo:

/// Compares JSON object for approximate equality.
/// You can use `[..]` wildcard in strings (useful for OS dependent things such
/// as paths). You can use a `"{...}"` string literal as a wildcard for
/// arbitrary nested JSON. Arrays are sorted before comparison.
fn find_mismatch<'a>(expected: &'a Value, actual: &'a Value) -> Option<(&'a Value, &'a Value)> {
    match (expected, actual) {
        (Value::Number(l), Value::Number(r)) if l == r => None,
        (Value::Bool(l), Value::Bool(r)) if l == r => None,
        (Value::String(l), Value::String(r)) if lines_match(l, r) => None,
        (Value::Array(l), Value::Array(r)) => {
            if l.len() != r.len() {
                return Some((expected, actual));
            }

            let mut l = l.iter().collect::<Vec<_>>();
            let mut r = r.iter().collect::<Vec<_>>();

            l.retain(|l| match r.iter().position(|r| find_mismatch(l, r).is_none()) {
                Some(i) => {
                    r.remove(i);
                    false
                }
                None => true,
            });

            if !l.is_empty() {
                assert!(!r.is_empty());
                Some((l[0], r[0]))
            } else {
                assert_eq!(r.len(), 0);
                None
            }
        }
        (Value::Object(l), Value::Object(r)) => {
            fn sorted_values(obj: &serde_json::Map<String, Value>) -> Vec<&Value> {
                let mut entries = obj.iter().collect::<Vec<_>>();
                entries.sort_by_key(|it| it.0);
                entries.into_iter().map(|(_k, v)| v).collect::<Vec<_>>()
            }

            let same_keys = l.len() == r.len() && l.keys().all(|k| r.contains_key(k));
            if !same_keys {
                return Some((expected, actual));
            }

            let l = sorted_values(l);
            let r = sorted_values(r);

            l.into_iter().zip(r).find_map(|(l, r)| find_mismatch(l, r))
        }
        (Value::Null, Value::Null) => None,
        // magic string literal "{...}" acts as wildcard for any sub-JSON
        (Value::String(l), _) if l == "{...}" => None,
        _ => Some((expected, actual)),
    }
}

/// Compare a line with an expected pattern.
/// - Use `[..]` as a wildcard to match 0 or more characters on the same line
///   (similar to `.*` in a regex).
fn lines_match(expected: &str, actual: &str) -> bool {
    // Let's not deal with / vs \ (windows...)
    // First replace backslash-escaped backslashes with forward slashes
    // which can occur in, for example, JSON output
    let expected = expected.replace(r"\\", "/").replace('\\', "/");
    let mut actual: &str = &actual.replace(r"\\", "/").replace('\\', "/");
    for (i, part) in expected.split("[..]").enumerate() {
        match actual.find(part) {
            Some(j) => {
                if i == 0 && j != 0 {
                    return false;
                }
                actual = &actual[j + part.len()..];
            }
            None => return false,
        }
    }
    actual.is_empty() || expected.ends_with("[..]")
}

#[test]
fn lines_match_works() {
    assert!(lines_match("a b", "a b"));
    assert!(lines_match("a[..]b", "a b"));
    assert!(lines_match("a[..]", "a b"));
    assert!(lines_match("[..]", "a b"));
    assert!(lines_match("[..]b", "a b"));

    assert!(!lines_match("[..]b", "c"));
    assert!(!lines_match("b", "c"));
    assert!(!lines_match("b", "cb"));
}
