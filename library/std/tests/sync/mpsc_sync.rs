use std::rc::Rc;
use std::sync::mpmc::SendTimeoutError;
use std::sync::mpsc::*;
use std::time::Duration;
use std::{env, thread};

pub fn stress_factor() -> usize {
    match env::var("RUST_TEST_STRESS") {
        Ok(val) => val.parse().unwrap(),
        Err(..) => 1,
    }
}

#[test]
fn smoke() {
    let (tx, rx) = sync_channel::<i32>(1);
    tx.send(1).unwrap();
    assert_eq!(rx.recv().unwrap(), 1);
}

#[test]
fn drop_full() {
    let (tx, _rx) = sync_channel::<Box<isize>>(1);
    tx.send(Box::new(1)).unwrap();
}

#[test]
fn smoke_shared() {
    let (tx, rx) = sync_channel::<i32>(1);
    tx.send(1).unwrap();
    assert_eq!(rx.recv().unwrap(), 1);
    let tx = tx.clone();
    tx.send(1).unwrap();
    assert_eq!(rx.recv().unwrap(), 1);
}

#[test]
fn recv_timeout() {
    let (tx, rx) = sync_channel::<i32>(1);
    assert_eq!(rx.recv_timeout(Duration::from_millis(1)), Err(RecvTimeoutError::Timeout));
    tx.send(1).unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_millis(1)), Ok(1));
}

#[test]
fn send_timeout() {
    let (tx, _rx) = sync_channel::<i32>(1);
    assert_eq!(tx.send_timeout(1, Duration::from_millis(1)), Ok(()));
    assert_eq!(tx.send_timeout(1, Duration::from_millis(1)), Err(SendTimeoutError::Timeout(1)));
}

#[test]
fn smoke_threads() {
    let (tx, rx) = sync_channel::<i32>(0);
    let _t = thread::spawn(move || {
        tx.send(1).unwrap();
    });
    assert_eq!(rx.recv().unwrap(), 1);
}

#[test]
fn smoke_port_gone() {
    let (tx, rx) = sync_channel::<i32>(0);
    drop(rx);
    assert!(tx.send(1).is_err());
}

#[test]
fn smoke_shared_port_gone2() {
    let (tx, rx) = sync_channel::<i32>(0);
    drop(rx);
    let tx2 = tx.clone();
    drop(tx);
    assert!(tx2.send(1).is_err());
}

#[test]
fn port_gone_concurrent() {
    let (tx, rx) = sync_channel::<i32>(0);
    let _t = thread::spawn(move || {
        rx.recv().unwrap();
    });
    while tx.send(1).is_ok() {}
}

#[test]
fn port_gone_concurrent_shared() {
    let (tx, rx) = sync_channel::<i32>(0);
    let tx2 = tx.clone();
    let _t = thread::spawn(move || {
        rx.recv().unwrap();
    });
    while tx.send(1).is_ok() && tx2.send(1).is_ok() {}
}

#[test]
fn smoke_chan_gone() {
    let (tx, rx) = sync_channel::<i32>(0);
    drop(tx);
    assert!(rx.recv().is_err());
}

#[test]
fn smoke_chan_gone_shared() {
    let (tx, rx) = sync_channel::<()>(0);
    let tx2 = tx.clone();
    drop(tx);
    drop(tx2);
    assert!(rx.recv().is_err());
}

#[test]
fn chan_gone_concurrent() {
    let (tx, rx) = sync_channel::<i32>(0);
    thread::spawn(move || {
        tx.send(1).unwrap();
        tx.send(1).unwrap();
    });
    while rx.recv().is_ok() {}
}

#[test]
fn stress() {
    let count = if cfg!(miri) { 100 } else { 10000 };
    let (tx, rx) = sync_channel::<i32>(0);
    thread::spawn(move || {
        for _ in 0..count {
            tx.send(1).unwrap();
        }
    });
    for _ in 0..count {
        assert_eq!(rx.recv().unwrap(), 1);
    }
}

#[test]
fn stress_recv_timeout_two_threads() {
    let count = if cfg!(miri) { 100 } else { 10000 };
    let (tx, rx) = sync_channel::<i32>(0);

    thread::spawn(move || {
        for _ in 0..count {
            tx.send(1).unwrap();
        }
    });

    let mut recv_count = 0;
    loop {
        match rx.recv_timeout(Duration::from_millis(1)) {
            Ok(v) => {
                assert_eq!(v, 1);
                recv_count += 1;
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    assert_eq!(recv_count, count);
}

#[test]
fn stress_recv_timeout_shared() {
    const AMT: u32 = if cfg!(miri) { 100 } else { 1000 };
    const NTHREADS: u32 = 8;
    let (tx, rx) = sync_channel::<i32>(0);
    let (dtx, drx) = sync_channel::<()>(0);

    thread::spawn(move || {
        let mut recv_count = 0;
        loop {
            match rx.recv_timeout(Duration::from_millis(10)) {
                Ok(v) => {
                    assert_eq!(v, 1);
                    recv_count += 1;
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }

        assert_eq!(recv_count, AMT * NTHREADS);
        assert!(rx.try_recv().is_err());

        dtx.send(()).unwrap();
    });

    for _ in 0..NTHREADS {
        let tx = tx.clone();
        thread::spawn(move || {
            for _ in 0..AMT {
                tx.send(1).unwrap();
            }
        });
    }

    drop(tx);

    drx.recv().unwrap();
}

#[test]
fn stress_shared() {
    const AMT: u32 = if cfg!(miri) { 100 } else { 1000 };
    const NTHREADS: u32 = 8;
    let (tx, rx) = sync_channel::<i32>(0);
    let (dtx, drx) = sync_channel::<()>(0);

    thread::spawn(move || {
        for _ in 0..AMT * NTHREADS {
            assert_eq!(rx.recv().unwrap(), 1);
        }
        match rx.try_recv() {
            Ok(..) => panic!(),
            _ => {}
        }
        dtx.send(()).unwrap();
    });

    for _ in 0..NTHREADS {
        let tx = tx.clone();
        thread::spawn(move || {
            for _ in 0..AMT {
                tx.send(1).unwrap();
            }
        });
    }
    drop(tx);
    drx.recv().unwrap();
}

#[test]
fn oneshot_single_thread_close_port_first() {
    // Simple test of closing without sending
    let (_tx, rx) = sync_channel::<i32>(0);
    drop(rx);
}

#[test]
fn oneshot_single_thread_close_chan_first() {
    // Simple test of closing without sending
    let (tx, _rx) = sync_channel::<i32>(0);
    drop(tx);
}

#[test]
fn oneshot_single_thread_send_port_close() {
    // Testing that the sender cleans up the payload if receiver is closed
    let (tx, rx) = sync_channel::<Box<i32>>(0);
    drop(rx);
    assert!(tx.send(Box::new(0)).is_err());
}

#[test]
fn oneshot_single_thread_recv_chan_close() {
    // Receiving on a closed chan will panic
    let res = thread::spawn(move || {
        let (tx, rx) = sync_channel::<i32>(0);
        drop(tx);
        rx.recv().unwrap();
    })
    .join();
    // What is our res?
    assert!(res.is_err());
}

#[test]
fn oneshot_single_thread_send_then_recv() {
    let (tx, rx) = sync_channel::<Box<i32>>(1);
    tx.send(Box::new(10)).unwrap();
    assert!(*rx.recv().unwrap() == 10);
}

#[test]
fn oneshot_single_thread_try_send_open() {
    let (tx, rx) = sync_channel::<i32>(1);
    assert_eq!(tx.try_send(10), Ok(()));
    assert!(rx.recv().unwrap() == 10);
}

#[test]
fn oneshot_single_thread_try_send_closed() {
    let (tx, rx) = sync_channel::<i32>(0);
    drop(rx);
    assert_eq!(tx.try_send(10), Err(TrySendError::Disconnected(10)));
}

#[test]
fn oneshot_single_thread_try_send_closed2() {
    let (tx, _rx) = sync_channel::<i32>(0);
    assert_eq!(tx.try_send(10), Err(TrySendError::Full(10)));
}

#[test]
fn oneshot_single_thread_try_recv_open() {
    let (tx, rx) = sync_channel::<i32>(1);
    tx.send(10).unwrap();
    assert!(rx.recv() == Ok(10));
}

#[test]
fn oneshot_single_thread_try_recv_closed() {
    let (tx, rx) = sync_channel::<i32>(0);
    drop(tx);
    assert!(rx.recv().is_err());
}

#[test]
fn oneshot_single_thread_try_recv_closed_with_data() {
    let (tx, rx) = sync_channel::<i32>(1);
    tx.send(10).unwrap();
    drop(tx);
    assert_eq!(rx.try_recv(), Ok(10));
    assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
}

#[test]
fn oneshot_single_thread_peek_data() {
    let (tx, rx) = sync_channel::<i32>(1);
    assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
    tx.send(10).unwrap();
    assert_eq!(rx.try_recv(), Ok(10));
}

#[test]
fn oneshot_single_thread_peek_close() {
    let (tx, rx) = sync_channel::<i32>(0);
    drop(tx);
    assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
    assert_eq!(rx.try_recv(), Err(TryRecvError::Disconnected));
}

#[test]
fn oneshot_single_thread_peek_open() {
    let (_tx, rx) = sync_channel::<i32>(0);
    assert_eq!(rx.try_recv(), Err(TryRecvError::Empty));
}

#[test]
fn oneshot_multi_task_recv_then_send() {
    let (tx, rx) = sync_channel::<Box<i32>>(0);
    let _t = thread::spawn(move || {
        assert!(*rx.recv().unwrap() == 10);
    });

    tx.send(Box::new(10)).unwrap();
}

#[test]
fn oneshot_multi_task_recv_then_close() {
    let (tx, rx) = sync_channel::<Box<i32>>(0);
    let _t = thread::spawn(move || {
        drop(tx);
    });
    let res = thread::spawn(move || {
        assert!(*rx.recv().unwrap() == 10);
    })
    .join();
    assert!(res.is_err());
}

#[test]
fn oneshot_multi_thread_close_stress() {
    for _ in 0..stress_factor() {
        let (tx, rx) = sync_channel::<i32>(0);
        let _t = thread::spawn(move || {
            drop(rx);
        });
        drop(tx);
    }
}

#[test]
fn oneshot_multi_thread_send_close_stress() {
    for _ in 0..stress_factor() {
        let (tx, rx) = sync_channel::<i32>(0);
        let _t = thread::spawn(move || {
            drop(rx);
        });
        let _ = thread::spawn(move || {
            tx.send(1).unwrap();
        })
        .join();
    }
}

#[test]
fn oneshot_multi_thread_recv_close_stress() {
    for _ in 0..stress_factor() {
        let (tx, rx) = sync_channel::<i32>(0);
        let _t = thread::spawn(move || {
            let res = thread::spawn(move || {
                rx.recv().unwrap();
            })
            .join();
            assert!(res.is_err());
        });
        let _t = thread::spawn(move || {
            thread::spawn(move || {
                drop(tx);
            });
        });
    }
}

#[test]
fn oneshot_multi_thread_send_recv_stress() {
    for _ in 0..stress_factor() {
        let (tx, rx) = sync_channel::<Box<i32>>(0);
        let _t = thread::spawn(move || {
            tx.send(Box::new(10)).unwrap();
        });
        assert!(*rx.recv().unwrap() == 10);
    }
}

#[test]
fn stream_send_recv_stress() {
    for _ in 0..stress_factor() {
        let (tx, rx) = sync_channel::<Box<i32>>(0);

        send(tx, 0);
        recv(rx, 0);

        fn send(tx: SyncSender<Box<i32>>, i: i32) {
            if i == 10 {
                return;
            }

            thread::spawn(move || {
                tx.send(Box::new(i)).unwrap();
                send(tx, i + 1);
            });
        }

        fn recv(rx: Receiver<Box<i32>>, i: i32) {
            if i == 10 {
                return;
            }

            thread::spawn(move || {
                assert!(*rx.recv().unwrap() == i);
                recv(rx, i + 1);
            });
        }
    }
}

#[test]
fn recv_a_lot() {
    let count = if cfg!(miri) { 1000 } else { 10000 };
    // Regression test that we don't run out of stack in scheduler context
    let (tx, rx) = sync_channel(count);
    for _ in 0..count {
        tx.send(()).unwrap();
    }
    for _ in 0..count {
        rx.recv().unwrap();
    }
}

#[test]
fn shared_chan_stress() {
    let (tx, rx) = sync_channel(0);
    let total = stress_factor() + 100;
    for _ in 0..total {
        let tx = tx.clone();
        thread::spawn(move || {
            tx.send(()).unwrap();
        });
    }

    for _ in 0..total {
        rx.recv().unwrap();
    }
}

#[test]
fn test_nested_recv_iter() {
    let (tx, rx) = sync_channel::<i32>(0);
    let (total_tx, total_rx) = sync_channel::<i32>(0);

    let _t = thread::spawn(move || {
        let mut acc = 0;
        for x in rx.iter() {
            acc += x;
        }
        total_tx.send(acc).unwrap();
    });

    tx.send(3).unwrap();
    tx.send(1).unwrap();
    tx.send(2).unwrap();
    drop(tx);
    assert_eq!(total_rx.recv().unwrap(), 6);
}

#[test]
fn test_recv_iter_break() {
    let (tx, rx) = sync_channel::<i32>(0);
    let (count_tx, count_rx) = sync_channel(0);

    let _t = thread::spawn(move || {
        let mut count = 0;
        for x in rx.iter() {
            if count >= 3 {
                break;
            } else {
                count += x;
            }
        }
        count_tx.send(count).unwrap();
    });

    tx.send(2).unwrap();
    tx.send(2).unwrap();
    tx.send(2).unwrap();
    let _ = tx.try_send(2);
    drop(tx);
    assert_eq!(count_rx.recv().unwrap(), 4);
}

#[test]
fn try_recv_states() {
    let (tx1, rx1) = sync_channel::<i32>(1);
    let (tx2, rx2) = sync_channel::<()>(1);
    let (tx3, rx3) = sync_channel::<()>(1);
    let _t = thread::spawn(move || {
        rx2.recv().unwrap();
        tx1.send(1).unwrap();
        tx3.send(()).unwrap();
        rx2.recv().unwrap();
        drop(tx1);
        tx3.send(()).unwrap();
    });

    assert_eq!(rx1.try_recv(), Err(TryRecvError::Empty));
    tx2.send(()).unwrap();
    rx3.recv().unwrap();
    assert_eq!(rx1.try_recv(), Ok(1));
    assert_eq!(rx1.try_recv(), Err(TryRecvError::Empty));
    tx2.send(()).unwrap();
    rx3.recv().unwrap();
    assert_eq!(rx1.try_recv(), Err(TryRecvError::Disconnected));
}

// This bug used to end up in a livelock inside of the Receiver destructor
// because the internal state of the Shared packet was corrupted
#[test]
fn destroy_upgraded_shared_port_when_sender_still_active() {
    let (tx, rx) = sync_channel::<()>(0);
    let (tx2, rx2) = sync_channel::<()>(0);
    let _t = thread::spawn(move || {
        rx.recv().unwrap(); // wait on a oneshot
        drop(rx); // destroy a shared
        tx2.send(()).unwrap();
    });
    // make sure the other thread has gone to sleep
    for _ in 0..5000 {
        thread::yield_now();
    }

    // upgrade to a shared chan and send a message
    let t = tx.clone();
    drop(tx);
    t.send(()).unwrap();

    // wait for the child thread to exit before we exit
    rx2.recv().unwrap();
}

#[test]
fn send1() {
    let (tx, rx) = sync_channel::<i32>(0);
    let _t = thread::spawn(move || {
        rx.recv().unwrap();
    });
    assert_eq!(tx.send(1), Ok(()));
}

#[test]
fn send2() {
    let (tx, rx) = sync_channel::<i32>(0);
    let _t = thread::spawn(move || {
        drop(rx);
    });
    assert!(tx.send(1).is_err());
}

#[test]
fn send3() {
    let (tx, rx) = sync_channel::<i32>(1);
    assert_eq!(tx.send(1), Ok(()));
    let _t = thread::spawn(move || {
        drop(rx);
    });
    assert!(tx.send(1).is_err());
}

#[test]
fn send4() {
    let (tx, rx) = sync_channel::<i32>(0);
    let tx2 = tx.clone();
    let (done, donerx) = channel();
    let done2 = done.clone();
    let _t = thread::spawn(move || {
        assert!(tx.send(1).is_err());
        done.send(()).unwrap();
    });
    let _t = thread::spawn(move || {
        assert!(tx2.send(2).is_err());
        done2.send(()).unwrap();
    });
    drop(rx);
    donerx.recv().unwrap();
    donerx.recv().unwrap();
}

#[test]
fn try_send1() {
    let (tx, _rx) = sync_channel::<i32>(0);
    assert_eq!(tx.try_send(1), Err(TrySendError::Full(1)));
}

#[test]
fn try_send2() {
    let (tx, _rx) = sync_channel::<i32>(1);
    assert_eq!(tx.try_send(1), Ok(()));
    assert_eq!(tx.try_send(1), Err(TrySendError::Full(1)));
}

#[test]
fn try_send3() {
    let (tx, rx) = sync_channel::<i32>(1);
    assert_eq!(tx.try_send(1), Ok(()));
    drop(rx);
    assert_eq!(tx.try_send(1), Err(TrySendError::Disconnected(1)));
}

#[test]
fn issue_15761() {
    fn repro() {
        let (tx1, rx1) = sync_channel::<()>(3);
        let (tx2, rx2) = sync_channel::<()>(3);

        let _t = thread::spawn(move || {
            rx1.recv().unwrap();
            tx2.try_send(()).unwrap();
        });

        tx1.try_send(()).unwrap();
        rx2.recv().unwrap();
    }

    for _ in 0..100 {
        repro()
    }
}

#[test]
fn drop_unreceived() {
    let (tx, rx) = sync_channel::<Rc<()>>(1);
    let msg = Rc::new(());
    let weak = Rc::downgrade(&msg);
    assert!(tx.send(msg).is_ok());
    drop(rx);
    // Messages should be dropped immediately when the last receiver is destroyed.
    assert!(weak.upgrade().is_none());
    drop(tx);
}
