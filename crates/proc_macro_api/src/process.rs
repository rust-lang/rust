//! Handle process life-time and message passing for proc-macro client

use std::{
    convert::{TryFrom, TryInto},
    ffi::{OsStr, OsString},
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Weak},
};

use crossbeam_channel::{bounded, Receiver, Sender};
use stdx::JodChild;

use crate::{
    msg::{ErrorCode, Message, Request, Response, ResponseError},
    rpc::{ListMacrosResult, ListMacrosTask, ProcMacroKind},
};

#[derive(Debug, Default)]
pub(crate) struct ProcMacroProcessSrv {
    inner: Weak<Sender<Task>>,
}

#[derive(Debug)]
pub(crate) struct ProcMacroProcessThread {
    // XXX: drop order is significant
    sender: Arc<Sender<Task>>,
    handle: jod_thread::JoinHandle<()>,
}

impl ProcMacroProcessSrv {
    pub(crate) fn run(
        process_path: PathBuf,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> io::Result<(ProcMacroProcessThread, ProcMacroProcessSrv)> {
        let process = Process::run(process_path, args)?;

        let (task_tx, task_rx) = bounded(0);
        let handle = jod_thread::spawn(move || {
            client_loop(task_rx, process);
        });

        let task_tx = Arc::new(task_tx);
        let srv = ProcMacroProcessSrv { inner: Arc::downgrade(&task_tx) };
        let thread = ProcMacroProcessThread { handle, sender: task_tx };

        Ok((thread, srv))
    }

    pub(crate) fn find_proc_macros(
        &self,
        dylib_path: &Path,
    ) -> Result<Vec<(String, ProcMacroKind)>, tt::ExpansionError> {
        let task = ListMacrosTask { lib: dylib_path.to_path_buf() };

        let result: ListMacrosResult = self.send_task(Request::ListMacro(task))?;
        Ok(result.macros)
    }

    pub(crate) fn send_task<R>(&self, req: Request) -> Result<R, tt::ExpansionError>
    where
        R: TryFrom<Response, Error = &'static str>,
    {
        let (result_tx, result_rx) = bounded(0);
        let sender = match self.inner.upgrade() {
            None => return Err(tt::ExpansionError::Unknown("proc macro process is closed".into())),
            Some(it) => it,
        };
        sender
            .send(Task { req, result_tx })
            .map_err(|_| tt::ExpansionError::Unknown("proc macro server crashed".into()))?;

        let res = result_rx
            .recv()
            .map_err(|_| tt::ExpansionError::Unknown("proc macro server crashed".into()))?;

        match res {
            Some(Response::Error(err)) => {
                return Err(tt::ExpansionError::ExpansionError(err.message));
            }
            Some(res) => Ok(res.try_into().map_err(|err| {
                tt::ExpansionError::Unknown(format!("Fail to get response, reason : {:#?} ", err))
            })?),
            None => Err(tt::ExpansionError::Unknown("Empty result".into())),
        }
    }
}

fn client_loop(task_rx: Receiver<Task>, mut process: Process) {
    let (mut stdin, mut stdout) = process.stdio().expect("couldn't access child stdio");

    let mut buf = String::new();

    for Task { req, result_tx } in task_rx {
        match send_request(&mut stdin, &mut stdout, req, &mut buf) {
            Ok(res) => result_tx.send(res).unwrap(),
            Err(err) => {
                log::error!(
                    "proc macro server crashed, server process state: {:?}, server request error: {:?}",
                    process.child.try_wait(),
                    err
                );
                let res = Response::Error(ResponseError {
                    code: ErrorCode::ServerErrorEnd,
                    message: "proc macro server crashed".into(),
                });
                result_tx.send(res.into()).unwrap();
                // Exit the thread.
                break;
            }
        }
    }
}

struct Task {
    req: Request,
    result_tx: Sender<Option<Response>>,
}

struct Process {
    child: JodChild,
}

impl Process {
    fn run(
        path: PathBuf,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> io::Result<Process> {
        let args: Vec<OsString> = args.into_iter().map(|s| s.as_ref().into()).collect();
        let child = JodChild(mk_child(&path, &args)?);
        Ok(Process { child })
    }

    fn stdio(&mut self) -> Option<(impl Write, impl BufRead)> {
        let stdin = self.child.stdin.take()?;
        let stdout = self.child.stdout.take()?;
        let read = BufReader::new(stdout);

        Some((stdin, read))
    }
}

fn mk_child(path: &Path, args: impl IntoIterator<Item = impl AsRef<OsStr>>) -> io::Result<Child> {
    Command::new(&path)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
}

fn send_request(
    mut writer: &mut impl Write,
    mut reader: &mut impl BufRead,
    req: Request,
    buf: &mut String,
) -> io::Result<Option<Response>> {
    req.write(&mut writer)?;
    Response::read(&mut reader, buf)
}
