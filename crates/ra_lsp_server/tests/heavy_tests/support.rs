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
    notification::{DidOpenTextDocument, Exit},
    request::Shutdown,
    ClientCapabilities, DidOpenTextDocumentParams, GotoCapability, TextDocumentClientCapabilities,
    TextDocumentIdentifier, TextDocumentItem, Url,
};
use serde::Serialize;
use serde_json::{to_string_pretty, Value};
use tempfile::TempDir;
use test_utils::{find_mismatch, parse_fixture};

use ra_lsp_server::{main_loop, req, ServerConfig};

pub struct Project<'a> {
    fixture: &'a str,
    with_sysroot: bool,
    tmp_dir: Option<TempDir>,
    roots: Vec<PathBuf>,
}

impl<'a> Project<'a> {
    pub fn with_fixture(fixture: &str) -> Project {
        Project { fixture, tmp_dir: None, roots: vec![], with_sysroot: false }
    }

    pub fn tmp_dir(mut self, tmp_dir: TempDir) -> Project<'a> {
        self.tmp_dir = Some(tmp_dir);
        self
    }

    pub fn root(mut self, path: &str) -> Project<'a> {
        self.roots.push(path.into());
        self
    }

    pub fn with_sysroot(mut self, sysroot: bool) -> Project<'a> {
        self.with_sysroot = sysroot;
        self
    }

    pub fn server(self) -> Server {
        let tmp_dir = self.tmp_dir.unwrap_or_else(|| TempDir::new().unwrap());
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = env_logger::builder().is_test(true).try_init().unwrap();
            ra_prof::set_filter(if crate::PROFILE.is_empty() {
                ra_prof::Filter::disabled()
            } else {
                ra_prof::Filter::from_spec(&crate::PROFILE)
            });
        });

        let mut paths = vec![];

        for entry in parse_fixture(self.fixture) {
            let path = tmp_dir.path().join(entry.meta);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path.as_path(), entry.text.as_bytes()).unwrap();
            paths.push((path, entry.text));
        }

        let roots = self.roots.into_iter().map(|root| tmp_dir.path().join(root)).collect();

        Server::new(tmp_dir, self.with_sysroot, roots, paths)
    }
}

pub fn project(fixture: &str) -> Server {
    Project::with_fixture(fixture).server()
}

pub struct Server {
    req_id: Cell<u64>,
    messages: RefCell<Vec<Message>>,
    dir: TempDir,
    _thread: jod_thread::JoinHandle<()>,
    client: Connection,
}

impl Server {
    fn new(
        dir: TempDir,
        with_sysroot: bool,
        roots: Vec<PathBuf>,
        files: Vec<(PathBuf, String)>,
    ) -> Server {
        let path = dir.path().to_path_buf();

        let roots = if roots.is_empty() { vec![path] } else { roots };
        let (connection, client) = Connection::memory();

        let _thread = jod_thread::Builder::new()
            .name("test server".to_string())
            .spawn(move || {
                main_loop(
                    roots,
                    ClientCapabilities {
                        workspace: None,
                        text_document: Some(TextDocumentClientCapabilities {
                            definition: Some(GotoCapability {
                                dynamic_registration: None,
                                link_support: Some(true),
                            }),
                            ..Default::default()
                        }),
                        window: None,
                        experimental: None,
                    },
                    ServerConfig { with_sysroot, ..ServerConfig::default() },
                    connection,
                )
                .unwrap()
            })
            .expect("failed to spawn a thread");

        let res =
            Server { req_id: Cell::new(1), dir, messages: Default::default(), client, _thread };

        for (path, text) in files {
            res.notification::<DidOpenTextDocument>(DidOpenTextDocumentParams {
                text_document: TextDocumentItem {
                    uri: Url::from_file_path(path).unwrap(),
                    language_id: "rust".to_string(),
                    version: 0,
                    text,
                },
            })
        }
        res
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
                Message::Request(req) => panic!("unexpected request: {:?}", req),
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
            Message::Notification(n) if n.method == "window/showMessage" => {
                let msg =
                    n.clone().extract::<req::ShowMessageParams>("window/showMessage").unwrap();
                msg.message.starts_with("workspace loaded")
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
