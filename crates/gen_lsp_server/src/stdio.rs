use std::{
    thread,
    io::{
        stdout, stdin,
    },
};

use crossbeam_channel::{Receiver, Sender, bounded};

use {RawMessage, Result};

pub fn stdio_transport() -> (Receiver<RawMessage>, Sender<RawMessage>, Threads) {
    let (writer_sender, mut writer_receiver) = bounded::<RawMessage>(16);
    let writer = thread::spawn(move || {
        let stdout = stdout();
        let mut stdout = stdout.lock();
        writer_receiver.try_for_each(|it| it.write(&mut stdout))?;
        Ok(())
    });
    let (reader_sender, reader_receiver) = bounded::<RawMessage>(16);
    let reader = thread::spawn(move || {
        let stdin = stdin();
        let mut stdin = stdin.lock();
        while let Some(msg) = RawMessage::read(&mut stdin)? {
            reader_sender.send(msg);
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
