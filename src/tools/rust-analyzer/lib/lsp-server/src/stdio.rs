use std::{
    io::{self, stdin, stdout},
    thread,
};

use log::debug;

use crossbeam_channel::{Receiver, Sender, bounded};

use crate::Message;

/// Creates an LSP connection via stdio.
pub(crate) fn stdio_transport() -> (Sender<Message>, Receiver<Message>, IoThreads) {
    let (drop_sender, drop_receiver) = bounded::<Message>(0);
    let (writer_sender, writer_receiver) = bounded::<Message>(0);
    let writer = thread::Builder::new()
        .name("LspServerWriter".to_owned())
        .spawn(move || {
            let stdout = stdout();
            let mut stdout = stdout.lock();
            writer_receiver.into_iter().try_for_each(|it| {
                let result = it.write(&mut stdout);
                let _ = drop_sender.send(it);
                result
            })
        })
        .unwrap();
    let dropper = thread::Builder::new()
        .name("LspMessageDropper".to_owned())
        .spawn(move || drop_receiver.into_iter().for_each(drop))
        .unwrap();
    let (reader_sender, reader_receiver) = bounded::<Message>(0);
    let reader = thread::Builder::new()
        .name("LspServerReader".to_owned())
        .spawn(move || {
            let stdin = stdin();
            let mut stdin = stdin.lock();
            while let Some(msg) = Message::read(&mut stdin)? {
                let is_exit = matches!(&msg, Message::Notification(n) if n.is_exit());

                debug!("sending message {msg:#?}");
                if let Err(e) = reader_sender.send(msg) {
                    return Err(io::Error::other(e));
                }

                if is_exit {
                    break;
                }
            }
            Ok(())
        })
        .unwrap();
    let threads = IoThreads { reader, writer, dropper };
    (writer_sender, reader_receiver, threads)
}

// Creates an IoThreads
pub(crate) fn make_io_threads(
    reader: thread::JoinHandle<io::Result<()>>,
    writer: thread::JoinHandle<io::Result<()>>,
    dropper: thread::JoinHandle<()>,
) -> IoThreads {
    IoThreads { reader, writer, dropper }
}

pub struct IoThreads {
    reader: thread::JoinHandle<io::Result<()>>,
    writer: thread::JoinHandle<io::Result<()>>,
    dropper: thread::JoinHandle<()>,
}

impl IoThreads {
    pub fn join(self) -> io::Result<()> {
        match self.reader.join() {
            Ok(r) => r?,
            Err(err) => std::panic::panic_any(err),
        }
        match self.dropper.join() {
            Ok(_) => (),
            Err(err) => {
                std::panic::panic_any(err);
            }
        }
        match self.writer.join() {
            Ok(r) => r,
            Err(err) => {
                std::panic::panic_any(err);
            }
        }
    }
}
