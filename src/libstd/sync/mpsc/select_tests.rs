#![allow(unused_imports)]

/// This file exists to hack around https://github.com/rust-lang/rust/issues/47238

use thread;
use sync::mpsc::*;

// Don't use the libstd version so we can pull in the right Select structure
// (std::comm points at the wrong one)
macro_rules! select {
    (
        $($name:pat = $rx:ident.$meth:ident() => $code:expr),+
    ) => ({
        let sel = Select::new();
        $( let mut $rx = sel.handle(&$rx); )+
        unsafe {
            $( $rx.add(); )+
        }
        let ret = sel.wait();
        $( if ret == $rx.id() { let $name = $rx.$meth(); $code } else )+
        { unreachable!() }
    })
}

#[test]
fn smoke() {
    let (tx1, rx1) = channel::<i32>();
    let (tx2, rx2) = channel::<i32>();
    tx1.send(1).unwrap();
    select! {
        foo = rx1.recv() => { assert_eq!(foo.unwrap(), 1); },
        _bar = rx2.recv() => { panic!() }
    }
    tx2.send(2).unwrap();
    select! {
        _foo = rx1.recv() => { panic!() },
        bar = rx2.recv() => { assert_eq!(bar.unwrap(), 2) }
    }
    drop(tx1);
    select! {
        foo = rx1.recv() => { assert!(foo.is_err()); },
        _bar = rx2.recv() => { panic!() }
    }
    drop(tx2);
    select! {
        bar = rx2.recv() => { assert!(bar.is_err()); }
    }
}

#[test]
fn smoke2() {
    let (_tx1, rx1) = channel::<i32>();
    let (_tx2, rx2) = channel::<i32>();
    let (_tx3, rx3) = channel::<i32>();
    let (_tx4, rx4) = channel::<i32>();
    let (tx5, rx5) = channel::<i32>();
    tx5.send(4).unwrap();
    select! {
        _foo = rx1.recv() => { panic!("1") },
        _foo = rx2.recv() => { panic!("2") },
        _foo = rx3.recv() => { panic!("3") },
        _foo = rx4.recv() => { panic!("4") },
        foo = rx5.recv() => { assert_eq!(foo.unwrap(), 4); }
    }
}

#[test]
fn closed() {
    let (_tx1, rx1) = channel::<i32>();
    let (tx2, rx2) = channel::<i32>();
    drop(tx2);

    select! {
        _a1 = rx1.recv() => { panic!() },
        a2 = rx2.recv() => { assert!(a2.is_err()); }
    }
}

#[test]
fn unblocks() {
    let (tx1, rx1) = channel::<i32>();
    let (_tx2, rx2) = channel::<i32>();
    let (tx3, rx3) = channel::<i32>();

    let _t = thread::spawn(move|| {
        for _ in 0..20 { thread::yield_now(); }
        tx1.send(1).unwrap();
        rx3.recv().unwrap();
        for _ in 0..20 { thread::yield_now(); }
    });

    select! {
        a = rx1.recv() => { assert_eq!(a.unwrap(), 1); },
        _b = rx2.recv() => { panic!() }
    }
    tx3.send(1).unwrap();
    select! {
        a = rx1.recv() => { assert!(a.is_err()) },
        _b = rx2.recv() => { panic!() }
    }
}

#[test]
fn both_ready() {
    let (tx1, rx1) = channel::<i32>();
    let (tx2, rx2) = channel::<i32>();
    let (tx3, rx3) = channel::<()>();

    let _t = thread::spawn(move|| {
        for _ in 0..20 { thread::yield_now(); }
        tx1.send(1).unwrap();
        tx2.send(2).unwrap();
        rx3.recv().unwrap();
    });

    select! {
        a = rx1.recv() => { assert_eq!(a.unwrap(), 1); },
        a = rx2.recv() => { assert_eq!(a.unwrap(), 2); }
    }
    select! {
        a = rx1.recv() => { assert_eq!(a.unwrap(), 1); },
        a = rx2.recv() => { assert_eq!(a.unwrap(), 2); }
    }
    assert_eq!(rx1.try_recv(), Err(TryRecvError::Empty));
    assert_eq!(rx2.try_recv(), Err(TryRecvError::Empty));
    tx3.send(()).unwrap();
}

#[test]
fn stress() {
    const AMT: i32 = 10000;
    let (tx1, rx1) = channel::<i32>();
    let (tx2, rx2) = channel::<i32>();
    let (tx3, rx3) = channel::<()>();

    let _t = thread::spawn(move|| {
        for i in 0..AMT {
            if i % 2 == 0 {
                tx1.send(i).unwrap();
            } else {
                tx2.send(i).unwrap();
            }
            rx3.recv().unwrap();
        }
    });

    for i in 0..AMT {
        select! {
            i1 = rx1.recv() => { assert!(i % 2 == 0 && i == i1.unwrap()); },
            i2 = rx2.recv() => { assert!(i % 2 == 1 && i == i2.unwrap()); }
        }
        tx3.send(()).unwrap();
    }
}

#[allow(unused_must_use)]
#[test]
fn cloning() {
    let (tx1, rx1) = channel::<i32>();
    let (_tx2, rx2) = channel::<i32>();
    let (tx3, rx3) = channel::<()>();

    let _t = thread::spawn(move|| {
        rx3.recv().unwrap();
        tx1.clone();
        assert_eq!(rx3.try_recv(), Err(TryRecvError::Empty));
        tx1.send(2).unwrap();
        rx3.recv().unwrap();
    });

    tx3.send(()).unwrap();
    select! {
        _i1 = rx1.recv() => {},
        _i2 = rx2.recv() => panic!()
    }
    tx3.send(()).unwrap();
}

#[allow(unused_must_use)]
#[test]
fn cloning2() {
    let (tx1, rx1) = channel::<i32>();
    let (_tx2, rx2) = channel::<i32>();
    let (tx3, rx3) = channel::<()>();

    let _t = thread::spawn(move|| {
        rx3.recv().unwrap();
        tx1.clone();
        assert_eq!(rx3.try_recv(), Err(TryRecvError::Empty));
        tx1.send(2).unwrap();
        rx3.recv().unwrap();
    });

    tx3.send(()).unwrap();
    select! {
        _i1 = rx1.recv() => {},
        _i2 = rx2.recv() => panic!()
    }
    tx3.send(()).unwrap();
}

#[test]
fn cloning3() {
    let (tx1, rx1) = channel::<()>();
    let (tx2, rx2) = channel::<()>();
    let (tx3, rx3) = channel::<()>();
    let _t = thread::spawn(move|| {
        let s = Select::new();
        let mut h1 = s.handle(&rx1);
        let mut h2 = s.handle(&rx2);
        unsafe { h2.add(); }
        unsafe { h1.add(); }
        assert_eq!(s.wait(), h2.id());
        tx3.send(()).unwrap();
    });

    for _ in 0..1000 { thread::yield_now(); }
    drop(tx1.clone());
    tx2.send(()).unwrap();
    rx3.recv().unwrap();
}

#[test]
fn preflight1() {
    let (tx, rx) = channel();
    tx.send(()).unwrap();
    select! {
        _n = rx.recv() => {}
    }
}

#[test]
fn preflight2() {
    let (tx, rx) = channel();
    tx.send(()).unwrap();
    tx.send(()).unwrap();
    select! {
        _n = rx.recv() => {}
    }
}

#[test]
fn preflight3() {
    let (tx, rx) = channel();
    drop(tx.clone());
    tx.send(()).unwrap();
    select! {
        _n = rx.recv() => {}
    }
}

#[test]
fn preflight4() {
    let (tx, rx) = channel();
    tx.send(()).unwrap();
    let s = Select::new();
    let mut h = s.handle(&rx);
    unsafe { h.add(); }
    assert_eq!(s.wait2(false), h.id());
}

#[test]
fn preflight5() {
    let (tx, rx) = channel();
    tx.send(()).unwrap();
    tx.send(()).unwrap();
    let s = Select::new();
    let mut h = s.handle(&rx);
    unsafe { h.add(); }
    assert_eq!(s.wait2(false), h.id());
}

#[test]
fn preflight6() {
    let (tx, rx) = channel();
    drop(tx.clone());
    tx.send(()).unwrap();
    let s = Select::new();
    let mut h = s.handle(&rx);
    unsafe { h.add(); }
    assert_eq!(s.wait2(false), h.id());
}

#[test]
fn preflight7() {
    let (tx, rx) = channel::<()>();
    drop(tx);
    let s = Select::new();
    let mut h = s.handle(&rx);
    unsafe { h.add(); }
    assert_eq!(s.wait2(false), h.id());
}

#[test]
fn preflight8() {
    let (tx, rx) = channel();
    tx.send(()).unwrap();
    drop(tx);
    rx.recv().unwrap();
    let s = Select::new();
    let mut h = s.handle(&rx);
    unsafe { h.add(); }
    assert_eq!(s.wait2(false), h.id());
}

#[test]
fn preflight9() {
    let (tx, rx) = channel();
    drop(tx.clone());
    tx.send(()).unwrap();
    drop(tx);
    rx.recv().unwrap();
    let s = Select::new();
    let mut h = s.handle(&rx);
    unsafe { h.add(); }
    assert_eq!(s.wait2(false), h.id());
}

#[test]
fn oneshot_data_waiting() {
    let (tx1, rx1) = channel();
    let (tx2, rx2) = channel();
    let _t = thread::spawn(move|| {
        select! {
            _n = rx1.recv() => {}
        }
        tx2.send(()).unwrap();
    });

    for _ in 0..100 { thread::yield_now() }
    tx1.send(()).unwrap();
    rx2.recv().unwrap();
}

#[test]
fn stream_data_waiting() {
    let (tx1, rx1) = channel();
    let (tx2, rx2) = channel();
    tx1.send(()).unwrap();
    tx1.send(()).unwrap();
    rx1.recv().unwrap();
    rx1.recv().unwrap();
    let _t = thread::spawn(move|| {
        select! {
            _n = rx1.recv() => {}
        }
        tx2.send(()).unwrap();
    });

    for _ in 0..100 { thread::yield_now() }
    tx1.send(()).unwrap();
    rx2.recv().unwrap();
}

#[test]
fn shared_data_waiting() {
    let (tx1, rx1) = channel();
    let (tx2, rx2) = channel();
    drop(tx1.clone());
    tx1.send(()).unwrap();
    rx1.recv().unwrap();
    let _t = thread::spawn(move|| {
        select! {
            _n = rx1.recv() => {}
        }
        tx2.send(()).unwrap();
    });

    for _ in 0..100 { thread::yield_now() }
    tx1.send(()).unwrap();
    rx2.recv().unwrap();
}

#[test]
fn sync1() {
    let (tx, rx) = sync_channel::<i32>(1);
    tx.send(1).unwrap();
    select! {
        n = rx.recv() => { assert_eq!(n.unwrap(), 1); }
    }
}

#[test]
fn sync2() {
    let (tx, rx) = sync_channel::<i32>(0);
    let _t = thread::spawn(move|| {
        for _ in 0..100 { thread::yield_now() }
        tx.send(1).unwrap();
    });
    select! {
        n = rx.recv() => { assert_eq!(n.unwrap(), 1); }
    }
}

#[test]
fn sync3() {
    let (tx1, rx1) = sync_channel::<i32>(0);
    let (tx2, rx2): (Sender<i32>, Receiver<i32>) = channel();
    let _t = thread::spawn(move|| { tx1.send(1).unwrap(); });
    let _t = thread::spawn(move|| { tx2.send(2).unwrap(); });
    select! {
        n = rx1.recv() => {
            let n = n.unwrap();
            assert_eq!(n, 1);
            assert_eq!(rx2.recv().unwrap(), 2);
        },
        n = rx2.recv() => {
            let n = n.unwrap();
            assert_eq!(n, 2);
            assert_eq!(rx1.recv().unwrap(), 1);
        }
    }
}
