use super::{Data, Empty, Inconsistent, Queue};
use crate::sync::mpsc::channel;
use crate::sync::Arc;
use crate::thread;

#[test]
fn test_full() {
    let q: Queue<Box<_>> = Queue::new();
    q.push(Box::new(1));
    q.push(Box::new(2));
}

#[test]
fn test() {
    let nthreads = 8;
    let nmsgs = 1000;
    let q = Queue::new();
    match q.pop() {
        Empty => {}
        Inconsistent | Data(..) => panic!(),
    }
    let (tx, rx) = channel();
    let q = Arc::new(q);

    for _ in 0..nthreads {
        let tx = tx.clone();
        let q = q.clone();
        thread::spawn(move || {
            for i in 0..nmsgs {
                q.push(i);
            }
            tx.send(()).unwrap();
        });
    }

    let mut i = 0;
    while i < nthreads * nmsgs {
        match q.pop() {
            Empty | Inconsistent => {}
            Data(_) => i += 1,
        }
    }
    drop(tx);
    for _ in 0..nthreads {
        rx.recv().unwrap();
    }
}
