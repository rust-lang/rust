use std::{
    io::{stdin, stdout},
    thread,
};

use crossbeam_channel::{bounded, Receiver, Sender};
use failure::bail;

use crate::{RawMessage, Result};

pub fn stdio_transport() -> (Receiver<RawMessage>, Sender<RawMessage>, Threads) {
    let (writer_sender, writer_receiver) = bounded::<RawMessage>(16);
    let writer = thread::spawn(move || {
        let stdout = stdout();
        let mut stdout = stdout.lock();
        writer_receiver.into_iter().try_for_each(|it| it.write(&mut stdout))?;
        Ok(())
    });
    let (reader_sender, reader_receiver) = bounded::<RawMessage>(16);
    let reader = thread::spawn(move || {
        let stdin = stdin();
        let mut stdin = stdin.lock();
        while let Some(msg) = RawMessage::read(&mut stdin)? {
            if let Err(_) = reader_sender.send(msg) {
                break;
            }
        }
        Ok(())
    });
    let threads = Threads { reader, writer };
    (reader_receiver, writer_sender, threads)
}

pub struct Threads {
    reader: thread::JoinHandle<Result<()>>,
    writer: thread::JoinHandle<Result<()>>,
}

impl Threads {
    pub fn join(self) -> Result<()> {
        match self.reader.join() {
            Ok(r) => r?,
            Err(_) => bail!("reader panicked"),
        }
        match self.writer.join() {
            Ok(r) => r,
            Err(_) => bail!("writer panicked"),
        }
    }
}
