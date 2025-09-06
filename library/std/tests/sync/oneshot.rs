//! Inspired by tests from <https://github.com/faern/oneshot/blob/main/tests/sync.rs>

use std::sync::mpsc::RecvError;
use std::sync::oneshot;
use std::sync::oneshot::{RecvTimeoutError, TryRecvError};
use std::time::{Duration, Instant};
use std::{mem, thread};

#[test]
fn send_before_try_recv() {
    let (sender, receiver) = oneshot::channel();

    assert!(sender.send(19i128).is_ok());

    match receiver.try_recv() {
        Ok(19) => {}
        _ => panic!("expected Ok(19)"),
    }
}

#[test]
fn send_before_recv() {
    let (sender, receiver) = oneshot::channel::<()>();

    assert!(sender.send(()).is_ok());
    assert_eq!(receiver.recv(), Ok(()));

    let (sender, receiver) = oneshot::channel::<u64>();

    assert!(sender.send(42).is_ok());
    assert_eq!(receiver.recv(), Ok(42));

    let (sender, receiver) = oneshot::channel::<[u8; 4096]>();

    assert!(sender.send([0b10101010; 4096]).is_ok());
    assert!(receiver.recv().unwrap()[..] == [0b10101010; 4096][..]);
}

#[test]
fn sender_drop() {
    {
        let (sender, receiver) = oneshot::channel::<u128>();

        mem::drop(sender);

        match receiver.recv() {
            Err(RecvError) => {}
            _ => panic!("expected recv error"),
        }
    }

    {
        let (sender, receiver) = oneshot::channel::<i32>();

        mem::drop(sender);

        match receiver.try_recv() {
            Err(TryRecvError::Disconnected) => {}
            _ => panic!("expected disconnected error"),
        }
    }
    {
        let (sender, receiver) = oneshot::channel::<i32>();

        mem::drop(sender);

        match receiver.recv_timeout(Duration::from_secs(1)) {
            Err(RecvTimeoutError::Disconnected) => {}
            _ => panic!("expected disconnected error"),
        }
    }
}

#[test]
fn send_never_deadline() {
    let (sender, receiver) = oneshot::channel::<i32>();

    mem::drop(sender);

    match receiver.recv_deadline(Instant::now()) {
        Err(RecvTimeoutError::Disconnected) => {}
        _ => panic!("expected disconnected error"),
    }
}

#[test]
fn send_before_recv_timeout() {
    let (sender, receiver) = oneshot::channel();

    assert!(sender.send(22i128).is_ok());

    let start = Instant::now();

    let timeout = Duration::from_secs(1);
    match receiver.recv_timeout(timeout) {
        Ok(22) => {}
        _ => panic!("expected Ok(22)"),
    }

    assert!(start.elapsed() < timeout);
}

#[test]
fn send_error() {
    let (sender, receiver) = oneshot::channel();

    mem::drop(receiver);

    let send_error = sender.send(32u128).unwrap_err();
    assert_eq!(send_error.0, 32);
}

#[test]
fn recv_before_send() {
    let (sender, receiver) = oneshot::channel();

    let t1 = thread::spawn(move || {
        thread::sleep(Duration::from_millis(10));
        sender.send(9u128).unwrap();
    });
    let t2 = thread::spawn(move || {
        assert_eq!(receiver.recv(), Ok(9));
    });

    t1.join().unwrap();
    t2.join().unwrap();
}

#[test]
fn recv_timeout_before_send() {
    let (sender, receiver) = oneshot::channel();

    let t = thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        sender.send(99u128).unwrap();
    });

    match receiver.recv_timeout(Duration::from_secs(1)) {
        Ok(99) => {}
        _ => panic!("expected Ok(99)"),
    }

    t.join().unwrap();
}

#[test]
fn recv_then_drop_sender() {
    let (sender, receiver) = oneshot::channel::<u128>();

    let t1 = thread::spawn(move || match receiver.recv() {
        Err(RecvError) => {}
        _ => panic!("expected recv error"),
    });

    let t2 = thread::spawn(move || {
        thread::sleep(Duration::from_millis(10));
        mem::drop(sender);
    });

    t1.join().unwrap();
    t2.join().unwrap();
}

#[test]
fn drop_sender_then_recv() {
    let (sender, receiver) = oneshot::channel::<u128>();

    let t1 = thread::spawn(move || {
        thread::sleep(Duration::from_millis(10));
        mem::drop(sender);
    });

    let t2 = thread::spawn(move || match receiver.recv() {
        Err(RecvError) => {}
        _ => panic!("expected disconnected error"),
    });

    t1.join().unwrap();
    t2.join().unwrap();
}

#[test]
fn try_recv_empty() {
    let (sender, receiver) = oneshot::channel::<u128>();
    match receiver.try_recv() {
        Err(TryRecvError::Empty(_)) => {}
        _ => panic!("expected empty error"),
    }
    mem::drop(sender);
}

#[test]
fn try_recv_then_drop_receiver() {
    let (sender, receiver) = oneshot::channel::<u128>();

    let t1 = thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        let _ = sender.send(42);
    });

    let t2 = thread::spawn(move || match receiver.try_recv() {
        Ok(_) => {}
        Err(TryRecvError::Empty(r)) => {
            mem::drop(r);
        }
        Err(TryRecvError::Disconnected) => {}
    });

    t2.join().unwrap();
    t1.join().unwrap();
}

#[test]
fn recv_no_time() {
    let (_sender, receiver) = oneshot::channel::<u128>();

    let start = Instant::now();
    match receiver.recv_deadline(start) {
        Err(RecvTimeoutError::Timeout(_)) => {}
        _ => panic!("expected timeout error"),
    }

    let (_sender, receiver) = oneshot::channel::<u128>();
    match receiver.recv_timeout(Duration::from_millis(0)) {
        Err(RecvTimeoutError::Timeout(_)) => {}
        _ => panic!("expected timeout error"),
    }
}

#[test]
fn recv_deadline_passed() {
    let (_sender, receiver) = oneshot::channel::<u128>();

    let start = Instant::now();
    let timeout = Duration::from_millis(100);

    match receiver.recv_deadline(start + timeout) {
        Err(RecvTimeoutError::Timeout(_)) => {}
        _ => panic!("expected timeout error"),
    }

    assert!(start.elapsed() >= timeout);
    assert!(start.elapsed() < timeout * 3);
}

#[test]
fn recv_time_passed() {
    let (_sender, receiver) = oneshot::channel::<u128>();

    let start = Instant::now();
    let timeout = Duration::from_millis(100);
    match receiver.recv_timeout(timeout) {
        Err(RecvTimeoutError::Timeout(_)) => {}
        _ => panic!("expected timeout error"),
    }
    assert!(start.elapsed() >= timeout);
    assert!(start.elapsed() < timeout * 3);
}

#[test]
fn non_send_type_can_be_used_on_same_thread() {
    use std::ptr;

    #[derive(Debug, Eq, PartialEq)]
    struct NotSend(*mut ());

    let (sender, receiver) = oneshot::channel();
    sender.send(NotSend(ptr::null_mut())).unwrap();
    let reply = receiver.try_recv().unwrap();
    assert_eq!(reply, NotSend(ptr::null_mut()));
}

/// Helper for testing drop behavior (taken directly from the `oneshot` crate).
struct DropCounter {
    count: std::rc::Rc<std::cell::RefCell<usize>>,
}

impl DropCounter {
    fn new() -> (DropTracker, DropCounter) {
        let count = std::rc::Rc::new(std::cell::RefCell::new(0));
        (DropTracker { count: count.clone() }, DropCounter { count })
    }

    fn count(&self) -> usize {
        *self.count.borrow()
    }
}

struct DropTracker {
    count: std::rc::Rc<std::cell::RefCell<usize>>,
}

impl Drop for DropTracker {
    fn drop(&mut self) {
        *self.count.borrow_mut() += 1;
    }
}

#[test]
fn message_in_channel_dropped_on_receiver_drop() {
    let (sender, receiver) = oneshot::channel();

    let (message, counter) = DropCounter::new();
    assert_eq!(counter.count(), 0);

    sender.send(message).unwrap();
    assert_eq!(counter.count(), 0);

    mem::drop(receiver);
    assert_eq!(counter.count(), 1);
}

#[test]
fn send_error_drops_message_correctly() {
    let (sender, receiver) = oneshot::channel();
    mem::drop(receiver);

    let (message, counter) = DropCounter::new();

    let send_error = sender.send(message).unwrap_err();
    assert_eq!(counter.count(), 0);

    mem::drop(send_error);
    assert_eq!(counter.count(), 1);
}

#[test]
fn send_error_drops_message_correctly_on_extract() {
    let (sender, receiver) = oneshot::channel();
    mem::drop(receiver);

    let (message, counter) = DropCounter::new();

    let send_error = sender.send(message).unwrap_err();
    assert_eq!(counter.count(), 0);

    let message = send_error.0; // Access the inner value directly
    assert_eq!(counter.count(), 0);

    mem::drop(message);
    assert_eq!(counter.count(), 1);
}
