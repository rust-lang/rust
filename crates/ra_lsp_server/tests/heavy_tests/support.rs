use std::{
    cell::{Cell, RefCell},
    fs,
    path::{Path, PathBuf},
    sync::Once,
    time::Duration,
};

use crossbeam_channel::{after, select, Receiver};
use flexi_logger::Logger;
use gen_lsp_server::{RawMessage, RawNotification, RawRequest};
use lsp_types::{
    notification::DidOpenTextDocument,
    notification::{Notification, ShowMessage},
    request::{Request, Shutdown},
    ClientCapabilities, DidOpenTextDocumentParams, GotoCapability, TextDocumentClientCapabilities,
    TextDocumentIdentifier, TextDocumentItem, Url,
};
use serde::Serialize;
use serde_json::{to_string_pretty, Value};
use tempfile::TempDir;
use test_utils::{find_mismatch, parse_fixture};
use thread_worker::Worker;

use ra_lsp_server::{main_loop, req, ServerConfig};

pub struct Project<'a> {
    fixture: &'a str,
    tmp_dir: Option<TempDir>,
    roots: Vec<PathBuf>,
}

impl<'a> Project<'a> {
    pub fn with_fixture(fixture: &str) -> Project {
        Project { fixture, tmp_dir: None, roots: vec![] }
    }

    pub fn tmp_dir(mut self, tmp_dir: TempDir) -> Project<'a> {
        self.tmp_dir = Some(tmp_dir);
        self
    }

    pub fn root(mut self, path: &str) -> Project<'a> {
        self.roots.push(path.into());
        self
    }

    pub fn server(self) -> Server {
        let tmp_dir = self.tmp_dir.unwrap_or_else(|| TempDir::new().unwrap());
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = Logger::with_env_or_str(crate::LOG).start().unwrap();
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

        Server::new(tmp_dir, roots, paths)
    }
}

pub fn project(fixture: &str) -> Server {
    Project::with_fixture(fixture).server()
}

pub struct Server {
    req_id: Cell<u64>,
    messages: RefCell<Vec<RawMessage>>,
    dir: TempDir,
    worker: Worker<RawMessage, RawMessage>,
}

impl Server {
    fn new(dir: TempDir, roots: Vec<PathBuf>, files: Vec<(PathBuf, String)>) -> Server {
        let path = dir.path().to_path_buf();

        let roots = if roots.is_empty() { vec![path] } else { roots };

        let worker = Worker::<RawMessage, RawMessage>::spawn(
            "test server",
            128,
            move |msg_receiver, msg_sender| {
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
                    ServerConfig::default(),
                    &msg_receiver,
                    &msg_sender,
                )
                .unwrap()
            },
        );
        let res = Server { req_id: Cell::new(1), dir, messages: Default::default(), worker };

        for (path, text) in files {
            res.send_notification(RawNotification::new::<DidOpenTextDocument>(
                &DidOpenTextDocumentParams {
                    text_document: TextDocumentItem {
                        uri: Url::from_file_path(path).unwrap(),
                        language_id: "rust".to_string(),
                        version: 0,
                        text,
                    },
                },
            ))
        }
        res
    }

    pub fn doc_id(&self, rel_path: &str) -> TextDocumentIdentifier {
        let path = self.dir.path().join(rel_path);
        TextDocumentIdentifier { uri: Url::from_file_path(path).unwrap() }
    }

    pub fn notification<N>(&self, params: N::Params)
    where
        N: Notification,
        N::Params: Serialize,
    {
        let r = RawNotification::new::<N>(&params);
        self.send_notification(r)
    }

    pub fn request<R>(&self, params: R::Params, expected_resp: Value)
    where
        R: Request,
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
        R: Request,
        R::Params: Serialize,
    {
        let id = self.req_id.get();
        self.req_id.set(id + 1);

        let r = RawRequest::new::<R>(id, &params);
        self.send_request_(r)
    }
    fn send_request_(&self, r: RawRequest) -> Value {
        let id = r.id;
        self.worker.sender().send(RawMessage::Request(r)).unwrap();
        while let Some(msg) = self.recv() {
            match msg {
                RawMessage::Request(req) => panic!("unexpected request: {:?}", req),
                RawMessage::Notification(_) => (),
                RawMessage::Response(res) => {
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
        self.wait_for_message_cond(1, &|msg: &RawMessage| match msg {
            RawMessage::Notification(n) if n.method == ShowMessage::METHOD => {
                let msg = n.clone().cast::<req::ShowMessage>().unwrap();
                msg.message.starts_with("workspace loaded")
            }
            _ => false,
        })
    }
    fn wait_for_message_cond(&self, n: usize, cond: &dyn Fn(&RawMessage) -> bool) {
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
    fn recv(&self) -> Option<RawMessage> {
        recv_timeout(&self.worker.receiver()).map(|msg| {
            self.messages.borrow_mut().push(msg.clone());
            msg
        })
    }
    fn send_notification(&self, not: RawNotification) {
        self.worker.sender().send(RawMessage::Notification(not)).unwrap();
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.send_request::<Shutdown>(());
    }
}

fn recv_timeout(receiver: &Receiver<RawMessage>) -> Option<RawMessage> {
    let timeout = Duration::from_secs(120);
    select! {
        recv(receiver) -> msg => msg.ok(),
        recv(after(timeout)) -> _ => panic!("timed out"),
    }
}
