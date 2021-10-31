use super::Builder;
use crate::any::Any;
use crate::mem;
use crate::result;
use crate::sync::{
    mpsc::{channel, Sender},
    Arc, Barrier,
};
use crate::thread::{self, ThreadId};
use crate::time::Duration;
use crate::time::Instant;

// !!! These tests are dangerous. If something is buggy, they will hang, !!!
// !!! instead of exiting cleanly. This might wedge the buildbots.       !!!

#[test]
fn test_unnamed_thread() {
    thread::spawn(move || {
        assert!(thread::current().name().is_none());
    })
    .join()
    .ok()
    .expect("thread panicked");
}

#[test]
fn test_named_thread() {
    Builder::new()
        .name("ada lovelace".to_string())
        .spawn(move || {
            assert!(thread::current().name().unwrap() == "ada lovelace".to_string());
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
#[should_panic]
fn test_invalid_named_thread() {
    let _ = Builder::new().name("ada l\0velace".to_string()).spawn(|| {});
}

#[test]
fn test_run_basic() {
    let (tx, rx) = channel();
    thread::spawn(move || {
        tx.send(()).unwrap();
    });
    rx.recv().unwrap();
}

#[test]
fn test_is_running() {
    let b = Arc::new(Barrier::new(2));
    let t = thread::spawn({
        let b = b.clone();
        move || {
            b.wait();
            1234
        }
    });

    // Thread is definitely running here, since it's still waiting for the barrier.
    assert_eq!(t.is_running(), true);

    // Unblock the barrier.
    b.wait();

    // Now check that t.is_running() becomes false within a reasonable time.
    let start = Instant::now();
    while t.is_running() {
        assert!(start.elapsed() < Duration::from_secs(2));
        thread::sleep(Duration::from_millis(15));
    }

    // Joining the thread should not block for a significant time now.
    let join_time = Instant::now();
    assert_eq!(t.join().unwrap(), 1234);
    assert!(join_time.elapsed() < Duration::from_secs(2));
}

#[test]
fn test_join_panic() {
    match thread::spawn(move || panic!()).join() {
        result::Result::Err(_) => (),
        result::Result::Ok(()) => panic!(),
    }
}

#[test]
fn test_spawn_sched() {
    let (tx, rx) = channel();

    fn f(i: i32, tx: Sender<()>) {
        let tx = tx.clone();
        thread::spawn(move || {
            if i == 0 {
                tx.send(()).unwrap();
            } else {
                f(i - 1, tx);
            }
        });
    }
    f(10, tx);
    rx.recv().unwrap();
}

#[test]
fn test_spawn_sched_childs_on_default_sched() {
    let (tx, rx) = channel();

    thread::spawn(move || {
        thread::spawn(move || {
            tx.send(()).unwrap();
        });
    });

    rx.recv().unwrap();
}

fn avoid_copying_the_body<F>(spawnfn: F)
where
    F: FnOnce(Box<dyn Fn() + Send>),
{
    let (tx, rx) = channel();

    let x: Box<_> = box 1;
    let x_in_parent = (&*x) as *const i32 as usize;

    spawnfn(Box::new(move || {
        let x_in_child = (&*x) as *const i32 as usize;
        tx.send(x_in_child).unwrap();
    }));

    let x_in_child = rx.recv().unwrap();
    assert_eq!(x_in_parent, x_in_child);
}

#[test]
fn test_avoid_copying_the_body_spawn() {
    avoid_copying_the_body(|v| {
        thread::spawn(move || v());
    });
}

#[test]
fn test_avoid_copying_the_body_thread_spawn() {
    avoid_copying_the_body(|f| {
        thread::spawn(move || {
            f();
        });
    })
}

#[test]
fn test_avoid_copying_the_body_join() {
    avoid_copying_the_body(|f| {
        let _ = thread::spawn(move || f()).join();
    })
}

#[test]
fn test_child_doesnt_ref_parent() {
    // If the child refcounts the parent thread, this will stack overflow when
    // climbing the thread tree to dereference each ancestor. (See #1789)
    // (well, it would if the constant were 8000+ - I lowered it to be more
    // valgrind-friendly. try this at home, instead..!)
    const GENERATIONS: u32 = 16;
    fn child_no(x: u32) -> Box<dyn Fn() + Send> {
        return Box::new(move || {
            if x < GENERATIONS {
                thread::spawn(move || child_no(x + 1)());
            }
        });
    }
    thread::spawn(|| child_no(0)());
}

#[test]
fn test_simple_newsched_spawn() {
    thread::spawn(move || {});
}

#[test]
fn test_try_panic_message_static_str() {
    match thread::spawn(move || {
        panic!("static string");
    })
    .join()
    {
        Err(e) => {
            type T = &'static str;
            assert!(e.is::<T>());
            assert_eq!(*e.downcast::<T>().unwrap(), "static string");
        }
        Ok(()) => panic!(),
    }
}

#[test]
fn test_try_panic_message_owned_str() {
    match thread::spawn(move || {
        panic!("owned string".to_string());
    })
    .join()
    {
        Err(e) => {
            type T = String;
            assert!(e.is::<T>());
            assert_eq!(*e.downcast::<T>().unwrap(), "owned string".to_string());
        }
        Ok(()) => panic!(),
    }
}

#[test]
fn test_try_panic_message_any() {
    match thread::spawn(move || {
        panic!(box 413u16 as Box<dyn Any + Send>);
    })
    .join()
    {
        Err(e) => {
            type T = Box<dyn Any + Send>;
            assert!(e.is::<T>());
            let any = e.downcast::<T>().unwrap();
            assert!(any.is::<u16>());
            assert_eq!(*any.downcast::<u16>().unwrap(), 413);
        }
        Ok(()) => panic!(),
    }
}

#[test]
fn test_try_panic_message_unit_struct() {
    struct Juju;

    match thread::spawn(move || panic!(Juju)).join() {
        Err(ref e) if e.is::<Juju>() => {}
        Err(_) | Ok(()) => panic!(),
    }
}

#[test]
fn test_park_timeout_unpark_before() {
    for _ in 0..10 {
        thread::current().unpark();
        thread::park_timeout(Duration::from_millis(u32::MAX as u64));
    }
}

#[test]
fn test_park_timeout_unpark_not_called() {
    for _ in 0..10 {
        thread::park_timeout(Duration::from_millis(10));
    }
}

#[test]
fn test_park_timeout_unpark_called_other_thread() {
    for _ in 0..10 {
        let th = thread::current();

        let _guard = thread::spawn(move || {
            super::sleep(Duration::from_millis(50));
            th.unpark();
        });

        thread::park_timeout(Duration::from_millis(u32::MAX as u64));
    }
}

#[test]
fn sleep_ms_smoke() {
    thread::sleep(Duration::from_millis(2));
}

#[test]
fn test_size_of_option_thread_id() {
    assert_eq!(mem::size_of::<Option<ThreadId>>(), mem::size_of::<ThreadId>());
}

#[test]
fn test_thread_id_equal() {
    assert!(thread::current().id() == thread::current().id());
}

#[test]
fn test_thread_id_not_equal() {
    let spawned_id = thread::spawn(|| thread::current().id()).join().unwrap();
    assert!(thread::current().id() != spawned_id);
}

// NOTE: the corresponding test for stderr is in ui/thread-stderr, due
// to the test harness apparently interfering with stderr configuration.
