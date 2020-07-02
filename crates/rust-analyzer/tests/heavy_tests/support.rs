use std::{
    cell::{Cell, RefCell},
    fs,
    path::{Path, PathBuf},
    sync::Once,
    time::Duration,
};

use crossbeam_channel::{after, select, Receiver};
use lsp_server::{Connection, Message, Notification, Request};
use lsp_types::{
    notification::Exit, request::Shutdown, TextDocumentIdentifier, Url, WorkDoneProgress,
};
use lsp_types::{ProgressParams, ProgressParamsValue};
use serde::Serialize;
use serde_json::{to_string_pretty, Value};
use tempfile::TempDir;
use test_utils::{find_mismatch, Fixture};

use ra_db::AbsPathBuf;
use ra_project_model::ProjectManifest;
use rust_analyzer::{
    config::{ClientCapsConfig, Config, FilesConfig, FilesWatcher, LinkedProject},
    main_loop,
};

pub struct Project<'a> {
    fixture: &'a str,
    with_sysroot: bool,
    tmp_dir: Option<TempDir>,
    roots: Vec<PathBuf>,
    config: Option<Box<dyn Fn(&mut Config)>>,
}

impl<'a> Project<'a> {
    pub fn with_fixture(fixture: &str) -> Project {
        Project { fixture, tmp_dir: None, roots: vec![], with_sysroot: false, config: None }
    }

    pub fn tmp_dir(mut self, tmp_dir: TempDir) -> Project<'a> {
        self.tmp_dir = Some(tmp_dir);
        self
    }

    pub(crate) fn root(mut self, path: &str) -> Project<'a> {
        self.roots.push(path.into());
        self
    }

    pub fn with_sysroot(mut self, sysroot: bool) -> Project<'a> {
        self.with_sysroot = sysroot;
        self
    }

    pub fn with_config(mut self, config: impl Fn(&mut Config) + 'static) -> Project<'a> {
        self.config = Some(Box::new(config));
        self
    }

    pub fn server(self) -> Server {
        let tmp_dir = self.tmp_dir.unwrap_or_else(|| TempDir::new().unwrap());
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            env_logger::builder().is_test(true).try_init().unwrap();
            ra_prof::init_from(crate::PROFILE);
        });

        for entry in Fixture::parse(self.fixture) {
            let path = tmp_dir.path().join(&entry.path['/'.len_utf8()..]);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path.as_path(), entry.text.as_bytes()).unwrap();
        }

        let tmp_dir_path = AbsPathBuf::assert(tmp_dir.path().to_path_buf());
        let mut roots =
            self.roots.into_iter().map(|root| tmp_dir_path.join(root)).collect::<Vec<_>>();
        if roots.is_empty() {
            roots.push(tmp_dir_path.clone());
        }
        let linked_projects = roots
            .into_iter()
            .map(|it| ProjectManifest::discover_single(&it).unwrap())
            .map(LinkedProject::from)
            .collect::<Vec<_>>();

        let mut config = Config {
            client_caps: ClientCapsConfig {
                location_link: true,
                code_action_literals: true,
                work_done_progress: true,
                ..Default::default()
            },
            with_sysroot: self.with_sysroot,
            linked_projects,
            files: FilesConfig { watcher: FilesWatcher::Client, exclude: Vec::new() },
            ..Config::new(tmp_dir_path)
        };
        if let Some(f) = &self.config {
            f(&mut config)
        }

        Server::new(tmp_dir, config)
    }
}

pub fn project(fixture: &str) -> Server {
    Project::with_fixture(fixture).server()
}

pub struct Server {
    req_id: Cell<u64>,
    messages: RefCell<Vec<Message>>,
    _thread: jod_thread::JoinHandle<()>,
    client: Connection,
    /// XXX: remove the tempdir last
    dir: TempDir,
}

impl Server {
    fn new(dir: TempDir, config: Config) -> Server {
        let (connection, client) = Connection::memory();

        let _thread = jod_thread::Builder::new()
            .name("test server".to_string())
            .spawn(move || main_loop(config, connection).unwrap())
            .expect("failed to spawn a thread");

        Server { req_id: Cell::new(1), dir, messages: Default::default(), client, _thread }
    }

    pub fn doc_id(&self, rel_path: &str) -> TextDocumentIdentifier {
        let path = self.dir.path().join(rel_path);
        TextDocumentIdentifier { uri: Url::from_file_path(path).unwrap() }
    }

    pub fn notification<N>(&self, params: N::Params)
    where
        N: lsp_types::notification::Notification,
        N::Params: Serialize,
    {
        let r = Notification::new(N::METHOD.to_string(), params);
        self.send_notification(r)
    }

    pub fn request<R>(&self, params: R::Params, expected_resp: Value)
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

    pub fn send_request<R>(&self, params: R::Params) -> Value
    where
        R: lsp_types::request::Request,
        R::Params: Serialize,
    {
        let id = self.req_id.get();
        self.req_id.set(id + 1);

        let r = Request::new(id.into(), R::METHOD.to_string(), params);
        self.send_request_(r)
    }
    fn send_request_(&self, r: Request) -> Value {
        let id = r.id.clone();
        self.client.sender.send(r.into()).unwrap();
        while let Some(msg) = self.recv() {
            match msg {
                Message::Request(req) => {
                    if req.method == "window/workDoneProgress/create" {
                        continue;
                    }
                    if req.method == "client/registerCapability" {
                        let params = req.params.to_string();
                        if ["workspace/didChangeWatchedFiles", "textDocument/didSave"]
                            .iter()
                            .any(|&it| params.contains(it))
                        {
                            continue;
                        }
                    }
                    panic!("unexpected request: {:?}", req)
                }
                Message::Notification(_) => (),
                Message::Response(res) => {
                    assert_eq!(res.id, id);
                    if let Some(err) = res.error {
                        panic!("error response: {:#?}", err);
                    }
                    return res.result.unwrap();
                }
            }
        }
        panic!("no response");
    }
    pub fn wait_until_workspace_is_loaded(&self) {
        self.wait_for_message_cond(1, &|msg: &Message| match msg {
            Message::Notification(n) if n.method == "$/progress" => {
                match n.clone().extract::<ProgressParams>("$/progress").unwrap() {
                    ProgressParams {
                        token: lsp_types::ProgressToken::String(ref token),
                        value: ProgressParamsValue::WorkDone(WorkDoneProgress::End(_)),
                    } if token == "rustAnalyzer/roots scanned" => true,
                    _ => false,
                }
            }
            _ => false,
        })
    }
    fn wait_for_message_cond(&self, n: usize, cond: &dyn Fn(&Message) -> bool) {
        let mut total = 0;
        for msg in self.messages.borrow().iter() {
            if cond(msg) {
                total += 1
            }
        }
        while total < n {
            let msg = self.recv().expect("no response");
            if cond(&msg) {
                total += 1;
            }
        }
    }
    fn recv(&self) -> Option<Message> {
        recv_timeout(&self.client.receiver).map(|msg| {
            self.messages.borrow_mut().push(msg.clone());
            msg
        })
    }
    fn send_notification(&self, not: Notification) {
        self.client.sender.send(Message::Notification(not)).unwrap();
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.request::<Shutdown>((), Value::Null);
        self.notification::<Exit>(());
    }
}

fn recv_timeout(receiver: &Receiver<Message>) -> Option<Message> {
    let timeout = Duration::from_secs(120);
    select! {
        recv(receiver) -> msg => msg.ok(),
        recv(after(timeout)) -> _ => panic!("timed out"),
    }
}
