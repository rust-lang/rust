use std::{
    fs,
    thread,
    cell::{Cell, RefCell},
    path::PathBuf,
    time::Duration,
    sync::Once,
};

use tempdir::TempDir;
use crossbeam_channel::{unbounded, after, Sender, Receiver};
use flexi_logger::Logger;
use languageserver_types::{
    Url,
    TextDocumentIdentifier,
    request::{Request, Shutdown},
    notification::DidOpenTextDocument,
    DidOpenTextDocumentParams,
    TextDocumentItem,
};
use serde::Serialize;
use serde_json::{Value, from_str, to_string_pretty};
use gen_lsp_server::{RawMessage, RawRequest, RawNotification};

use m::{Result, main_loop, req};

pub fn project(fixture: &str) -> Server {
    static INIT: Once = Once::new();
    INIT.call_once(|| Logger::with_env_or_str(::LOG).start().unwrap());

    let tmp_dir = TempDir::new("test-project")
        .unwrap();
    let mut buf = String::new();
    let mut file_name = None;
    let mut paths = vec![];
    macro_rules! flush {
        () => {
            if let Some(file_name) = file_name {
                let path = tmp_dir.path().join(file_name);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(path.as_path(), buf.as_bytes()).unwrap();
                paths.push((path, buf.clone()));
            }
        }
    };
    for line in fixture.lines() {
        if line.starts_with("//-") {
            flush!();
            buf.clear();
            file_name = Some(line["//-".len()..].trim());
            continue;
        }
        buf.push_str(line);
        buf.push('\n');
    }
    flush!();
    Server::new(tmp_dir, paths)
}

pub struct Server {
    req_id: Cell<u64>,
    messages: RefCell<Vec<RawMessage>>,
    dir: TempDir,
    sender: Option<Sender<RawMessage>>,
    receiver: Receiver<RawMessage>,
    server: Option<thread::JoinHandle<Result<()>>>,
}

impl Server {
    fn new(dir: TempDir, files: Vec<(PathBuf, String)>) -> Server {
        let path = dir.path().to_path_buf();
        let (client_sender, mut server_receiver) = unbounded();
        let (mut server_sender, client_receiver) = unbounded();
        let server = thread::spawn(move || main_loop(true, path, &mut server_receiver, &mut server_sender));
        let res = Server {
            req_id: Cell::new(1),
            dir,
            messages: Default::default(),
            sender: Some(client_sender),
            receiver: client_receiver,
            server: Some(server),
        };

        for (path, text) in files {
            res.send_notification(RawNotification::new::<DidOpenTextDocument>(
                &DidOpenTextDocumentParams {
                    text_document: TextDocumentItem {
                        uri: Url::from_file_path(path).unwrap(),
                        language_id: "rust".to_string(),
                        version: 0,
                        text,
                    }
                }
            ))
        }
        res
    }

    pub fn doc_id(&self, rel_path: &str) -> TextDocumentIdentifier {
        let path = self.dir.path().join(rel_path);
        TextDocumentIdentifier {
            uri: Url::from_file_path(path).unwrap(),
        }
    }

    pub fn request<R>(
        &self,
        params: R::Params,
        expected_resp: &str,
    )
    where
        R: Request,
        R::Params: Serialize,
    {
        let id = self.req_id.get();
        self.req_id.set(id + 1);
        let expected_resp: Value = from_str(expected_resp).unwrap();
        let actual = self.send_request::<R>(id, params);
        assert_eq!(
            expected_resp, actual,
            "Expected:\n{}\n\
             Actual:\n{}\n",
            to_string_pretty(&expected_resp).unwrap(),
            to_string_pretty(&actual).unwrap(),
        );
    }

    fn send_request<R>(&self, id: u64, params: R::Params) -> Value
    where
        R: Request,
        R::Params: Serialize,
    {
        let r = RawRequest::new::<R>(id, &params);
        self.send_request_(r)
    }
    fn send_request_(&self, r: RawRequest) -> Value
    {
        let id = r.id;
        self.sender.as_ref()
            .unwrap()
            .send(RawMessage::Request(r));
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
    pub fn wait_for_feedback(&self, feedback: &str) {
        self.wait_for_feedback_n(feedback, 1)
    }
    pub fn wait_for_feedback_n(&self, feedback: &str, n: usize) {
        let f = |msg: &RawMessage| match msg {
                RawMessage::Notification(n) if n.method == "internalFeedback" => {
                    return n.clone().cast::<req::InternalFeedback>()
                        .unwrap() == feedback
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
        recv_timeout(&self.receiver)
            .map(|msg| {
                self.messages.borrow_mut().push(msg.clone());
                msg
            })
    }
    fn send_notification(&self, not: RawNotification) {
        self.sender.as_ref()
            .unwrap()
            .send(RawMessage::Notification(not));
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        {
            self.send_request::<Shutdown>(666, ());
            drop(self.sender.take().unwrap());
            while let Some(msg) = recv_timeout(&self.receiver) {
                drop(msg);
            }
        }
        self.server.take()
            .unwrap()
            .join().unwrap().unwrap();
    }
}

fn recv_timeout(receiver: &Receiver<RawMessage>) -> Option<RawMessage> {
    let timeout = Duration::from_secs(5);
    select! {
        recv(receiver, msg) => msg,
        recv(after(timeout)) => panic!("timed out"),
    }
}
