use std::{
    io::{self, BufReader},
    net::TcpStream,
    thread,
};

use crossbeam_channel::{Receiver, Sender, bounded};

use crate::{
    Message,
    stdio::{IoThreads, make_io_threads},
};

pub(crate) fn socket_transport(
    stream: TcpStream,
) -> (Sender<Message>, Receiver<Message>, IoThreads) {
    let (reader_receiver, reader) = make_reader(stream.try_clone().unwrap());
    let (writer_sender, writer, messages_to_drop) = make_write(stream);
    let dropper = std::thread::spawn(move || {
        messages_to_drop.into_iter().for_each(drop);
    });
    let io_threads = make_io_threads(reader, writer, dropper);
    (writer_sender, reader_receiver, io_threads)
}

fn make_reader(stream: TcpStream) -> (Receiver<Message>, thread::JoinHandle<io::Result<()>>) {
    let (reader_sender, reader_receiver) = bounded::<Message>(0);
    let reader = thread::spawn(move || {
        let mut buf_read = BufReader::new(stream);
        while let Some(msg) = Message::read(&mut buf_read).unwrap() {
            let is_exit = matches!(&msg, Message::Notification(n) if n.is_exit());
            reader_sender.send(msg).unwrap();
            if is_exit {
                break;
            }
        }
        Ok(())
    });
    (reader_receiver, reader)
}

fn make_write(
    mut stream: TcpStream,
) -> (Sender<Message>, thread::JoinHandle<io::Result<()>>, Receiver<Message>) {
    let (writer_sender, writer_receiver) = bounded::<Message>(0);
    let (drop_sender, drop_receiver) = bounded::<Message>(0);
    let writer = thread::spawn(move || {
        writer_receiver
            .into_iter()
            .try_for_each(|it| {
                let result = it.write(&mut stream);
                let _ = drop_sender.send(it);
                result
            })
            .unwrap();
        Ok(())
    });
    (writer_sender, writer, drop_receiver)
}
