use std::{
    cell::{Cell, RefCell},
    fs,
    path::PathBuf,
    sync::Once,
    time::Duration,
};

use crossbeam_channel::{after, select, Receiver};
use flexi_logger::Logger;
use gen_lsp_server::{RawMessage, RawNotification, RawRequest};
use lsp_types::{
    notification::DidOpenTextDocument,
    request::{Request, Shutdown},
    DidOpenTextDocumentParams, TextDocumentIdentifier, TextDocumentItem, Url,
    notification::{Notification, ShowMessage},
};
use serde::Serialize;
use serde_json::{to_string_pretty, Value};
use tempfile::TempDir;
use thread_worker::Worker;
use test_utils::{parse_fixture, find_mismatch};

use ra_lsp_server::{
    main_loop, req,
    InitializationOptions,
};

pub fn project(fixture: &str) -> Server {
    let tmp_dir = TempDir::new().unwrap();
    project_with_tmpdir(tmp_dir, fixture)
}

pub fn project_with_tmpdir(tmp_dir: TempDir, fixture: &str) -> Server {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = Logger::with_env_or_str(crate::LOG).start().unwrap();
    });

    let mut paths = vec![];

    for entry in parse_fixture(fixture) {
        let path = tmp_dir.path().join(entry.meta);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path.as_path(), entry.text.as_bytes()).unwrap();
        paths.push((path, entry.text));
    }
    Server::new(tmp_dir, paths)
}

pub struct Server {
    req_id: Cell<u64>,
    messages: RefCell<Vec<RawMessage>>,
    dir: TempDir,
    worker: Option<Worker<RawMessage, RawMessage>>,
}

impl Server {
    fn new(dir: TempDir, files: Vec<(PathBuf, String)>) -> Server {
        let path = dir.path().to_path_buf();
        let worker = Worker::<RawMessage, RawMessage>::spawn(
            "test server",
            128,
            move |mut msg_receiver, mut msg_sender| {
                main_loop(
                    path,
                    InitializationOptions::default(),
                    &mut msg_receiver,
                    &mut msg_sender,
                )
                .unwrap()
            },
        );
        let res = Server {
            req_id: Cell::new(1),
            dir,
            messages: Default::default(),
            worker: Some(worker),
        };

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

    pub fn request<R>(&self, params: R::Params, expected_resp: Value)
    where
        R: Request,
        R::Params: Serialize,
    {
        let actual = self.send_request::<R>(params);
        match find_mismatch(&expected_resp, &actual) {
            Some((expected_part, actual_part)) => panic!(
                "JSON mismatch\nExpected:\n{}\nWas:\n{}\nExpected part:\n{}\nActual part:\n{}\n",
                to_string_pretty(&expected_resp).unwrap(),
                to_string_pretty(&actual).unwrap(),
                to_string_pretty(expected_part).unwrap(),
                to_string_pretty(actual_part).unwrap(),
            ),
            None => {}
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
        self.worker.as_ref().unwrap().sender().send(RawMessage::Request(r)).unwrap();
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
    pub fn wait_for_message(&self, message: &str) {
        self.wait_for_message_n(message, 1)
    }
    pub fn wait_for_message_n(&self, message: &str, n: usize) {
        let f = |msg: &RawMessage| match msg {
            RawMessage::Notification(n) if n.method == ShowMessage::METHOD => {
                let msg = n.clone().cast::<req::ShowMessage>().unwrap();
                msg.message == message
            }
            _ => false,
        };
        let mut total = 0;
        for msg in self.messages.borrow().iter() {
            if f(msg) {
                total += 1
            }
        }
        while total < n {
            let msg = self.recv().expect("no response");
            if f(&msg) {
                total += 1;
            }
        }
    }
    fn recv(&self) -> Option<RawMessage> {
        recv_timeout(&self.worker.as_ref().unwrap().receiver()).map(|msg| {
            self.messages.borrow_mut().push(msg.clone());
            msg
        })
    }
    fn send_notification(&self, not: RawNotification) {
        self.worker.as_ref().unwrap().sender().send(RawMessage::Notification(not)).unwrap();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.send_request::<Shutdown>(());
    }
}

fn recv_timeout(receiver: &Receiver<RawMessage>) -> Option<RawMessage> {
    let timeout = Duration::from_secs(50);
    select! {
        recv(receiver) -> msg => msg.ok(),
        recv(after(timeout)) -> _ => panic!("timed out"),
    }
}
