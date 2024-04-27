use std::sync::mpsc::channel;

pub fn main() {
    let (tx, rx) = channel::<Box<_>>();
    tx.send(Box::new(100)).unwrap();
    let v = rx.recv().unwrap();
    assert_eq!(v, Box::new(100));

    tx.send(Box::new(101)).unwrap();
    tx.send(Box::new(102)).unwrap();
    assert_eq!(rx.recv().unwrap(), Box::new(101));
    assert_eq!(rx.recv().unwrap(), Box::new(102));
}
