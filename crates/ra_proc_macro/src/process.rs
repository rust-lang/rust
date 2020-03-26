use crossbeam_channel::{bounded, Receiver, Sender};
use ra_tt::Subtree;

use crate::msg::{ErrorCode, Message, Request, Response, ResponseError};
use crate::rpc::{ExpansionResult, ExpansionTask, ListMacrosResult, ListMacrosTask, ProcMacroKind};

use io::{BufRead, BufReader};
use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread::spawn,
};

#[derive(Debug, Default)]
pub(crate) struct ProcMacroProcessSrv {
    inner: Option<Handle>,
}

struct Task {
    req: Message,
    result_tx: Sender<Message>,
}

#[derive(Debug)]
struct Handle {
    sender: Sender<Task>,
}

struct Process {
    path: PathBuf,
    child: Child,
}

impl Process {
    fn run(process_path: &Path) -> Result<Process, io::Error> {
        let child = Command::new(process_path.clone())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;

        Ok(Process { path: process_path.into(), child })
    }

    fn restart(&mut self) -> Result<(), io::Error> {
        let _ = self.child.kill();
        self.child =
            Command::new(self.path.clone()).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
        Ok(())
    }

    fn stdio(&mut self) -> Option<(impl Write, impl BufRead)> {
        let stdin = self.child.stdin.take()?;
        let stdout = self.child.stdout.take()?;
        let read = BufReader::new(stdout);

        Some((stdin, read))
    }
}

impl ProcMacroProcessSrv {
    pub fn run(process_path: &Path) -> Result<ProcMacroProcessSrv, io::Error> {
        let process = Process::run(process_path)?;

        let (task_tx, task_rx) = bounded(0);

        let _ = spawn(move || {
            client_loop(task_rx, process);
        });
        Ok(ProcMacroProcessSrv { inner: Some(Handle { sender: task_tx }) })
    }

    pub fn find_proc_macros(
        &self,
        dylib_path: &Path,
    ) -> Result<Vec<(String, ProcMacroKind)>, ra_tt::ExpansionError> {
        let task = ListMacrosTask { lib: dylib_path.to_path_buf() };

        let result: ListMacrosResult = self.send_task("list_macros", task)?;
        Ok(result.macros)
    }

    pub fn custom_derive(
        &self,
        dylib_path: &Path,
        subtree: &Subtree,
        derive_name: &str,
    ) -> Result<Subtree, ra_tt::ExpansionError> {
        let task = ExpansionTask {
            macro_body: subtree.clone(),
            macro_name: derive_name.to_string(),
            attributes: None,
            lib: dylib_path.to_path_buf(),
        };

        let result: ExpansionResult = self.send_task("custom_derive", task)?;
        Ok(result.expansion)
    }

    pub fn send_task<'a, T, R>(&self, method: &str, task: T) -> Result<R, ra_tt::ExpansionError>
    where
        T: serde::Serialize,
        R: serde::de::DeserializeOwned + Default,
    {
        let handle = match &self.inner {
            None => return Err(ra_tt::ExpansionError::Unknown("No handle is found.".to_string())),
            Some(it) => it,
        };

        let msg = serde_json::to_value(task).unwrap();

        // FIXME: use a proper request id
        let id = 0;
        let req = Request { id: id.into(), method: method.into(), params: msg };

        let (result_tx, result_rx) = bounded(0);

        handle.sender.send(Task { req: req.into(), result_tx }).unwrap();
        let response = result_rx.recv().unwrap();

        match response {
            Message::Request(_) => {
                return Err(ra_tt::ExpansionError::Unknown(
                    "Return request from ra_proc_srv".into(),
                ))
            }
            Message::Response(res) => {
                if let Some(err) = res.error {
                    return Err(ra_tt::ExpansionError::ExpansionError(err.message));
                }
                match res.result {
                    None => Ok(R::default()),
                    Some(res) => {
                        let result: R = serde_json::from_value(res)
                            .map_err(|err| ra_tt::ExpansionError::JsonError(err.to_string()))?;
                        Ok(result)
                    }
                }
            }
        }
    }
}

fn client_loop(task_rx: Receiver<Task>, mut process: Process) {
    let (mut stdin, mut stdout) = match process.stdio() {
        None => return,
        Some(it) => it,
    };

    loop {
        let task = match task_rx.recv() {
            Ok(task) => task,
            Err(_) => break,
        };

        let res = match send_message(&mut stdin, &mut stdout, task.req) {
            Ok(res) => res,
            Err(_err) => {
                let res = Response {
                    id: 0.into(),
                    result: None,
                    error: Some(ResponseError {
                        code: ErrorCode::ServerErrorEnd as i32,
                        message: "Server closed".into(),
                        data: None,
                    }),
                };
                if task.result_tx.send(res.into()).is_err() {
                    break;
                }
                // Restart the process
                if process.restart().is_err() {
                    break;
                }
                let stdio = match process.stdio() {
                    None => break,
                    Some(it) => it,
                };
                stdin = stdio.0;
                stdout = stdio.1;
                continue;
            }
        };

        if let Some(res) = res {
            if task.result_tx.send(res).is_err() {
                break;
            }
        }
    }

    let _ = process.child.kill();
}

fn send_message(
    mut writer: &mut impl Write,
    mut reader: &mut impl BufRead,
    msg: Message,
) -> Result<Option<Message>, io::Error> {
    msg.write(&mut writer)?;
    Ok(Message::read(&mut reader)?)
}
