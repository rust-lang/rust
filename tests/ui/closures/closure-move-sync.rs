use std::thread;
use std::sync::mpsc::channel;

fn bar() {
    let (send, recv) = channel();
    let t = thread::spawn(|| {
        recv.recv().unwrap();
        //~^^ ERROR `std::sync::mpsc::Receiver<()>` cannot be shared between threads safely
    });

    send.send(());

    t.join().unwrap();
}

fn main() {}
