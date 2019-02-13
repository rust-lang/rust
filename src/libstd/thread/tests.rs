#[cfg(not(target_os = "emscripten"))]
mod thread {
    use any::Any;
    use sync::mpsc::{channel, Sender};
    use result;
    use thread::{self, Builder};
    use time::Duration;
    use u32;

    // !!! These tests are dangerous. If something is buggy, they will hang, !!!
    // !!! instead of exiting cleanly. This might wedge the buildbots.       !!!

    #[test]
    fn test_unnamed_thread() {
        thread::spawn(move|| {
            assert!(thread::current().name().is_none());
        }).join().ok().unwrap();
    }

    #[test]
    fn test_named_thread() {
        Builder::new().name("ada lovelace".to_string()).spawn(move|| {
            assert!(thread::current().name().unwrap() == "ada lovelace".to_string());
        }).unwrap().join().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_invalid_named_thread() {
        let _ = Builder::new().name("ada l\0velace".to_string()).spawn(|| {});
    }

    #[test]
    fn test_run_basic() {
        let (tx, rx) = channel();
        thread::spawn(move|| {
            tx.send(()).unwrap();
        });
        rx.recv().unwrap();
    }

    #[test]
    fn test_join_panic() {
        match thread::spawn(move|| {
            panic!()
        }).join() {
            result::Result::Err(_) => (),
            result::Result::Ok(()) => panic!()
        }
    }

    #[test]
    fn test_spawn_sched() {
        let (tx, rx) = channel();

        fn f(i: i32, tx: Sender<()>) {
            let tx = tx.clone();
            thread::spawn(move|| {
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

        thread::spawn(move|| {
            thread::spawn(move|| {
                tx.send(()).unwrap();
            });
        });

        rx.recv().unwrap();
    }

    fn avoid_copying_the_body<F>(spawnfn: F) where F: FnOnce(Box<dyn Fn() + Send>) {
        let (tx, rx) = channel();

        let x: Box<_> = box 1;
        let x_in_parent = (&*x) as *const i32 as usize;

        spawnfn(Box::new(move|| {
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
            thread::spawn(move|| {
                f();
            });
        })
    }

    #[test]
    fn test_avoid_copying_the_body_join() {
        avoid_copying_the_body(|f| {
            let _ = thread::spawn(move|| {
                f()
            }).join();
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
            return Box::new(move|| {
                if x < GENERATIONS {
                    thread::spawn(move|| child_no(x+1)());
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
        match thread::spawn(move|| {
            panic!("static string");
        }).join() {
            Err(e) => {
                type T = &'static str;
                assert!(e.is::<T>());
                assert_eq!(*e.downcast::<T>().unwrap(), "static string");
            }
            Ok(()) => panic!()
        }
    }

    #[test]
    fn test_try_panic_message_owned_str() {
        match thread::spawn(move|| {
            panic!("owned string".to_string());
        }).join() {
            Err(e) => {
                type T = String;
                assert!(e.is::<T>());
                assert_eq!(*e.downcast::<T>().unwrap(), "owned string".to_string());
            }
            Ok(()) => panic!()
        }
    }

    #[test]
    fn test_try_panic_message_any() {
        match thread::spawn(move|| {
            panic!(box 413u16 as Box<dyn Any + Send>);
        }).join() {
            Err(e) => {
                type T = Box<dyn Any + Send>;
                assert!(e.is::<T>());
                let any = e.downcast::<T>().unwrap();
                assert!(any.is::<u16>());
                assert_eq!(*any.downcast::<u16>().unwrap(), 413);
            }
            Ok(()) => panic!()
        }
    }

    #[test]
    fn test_try_panic_message_unit_struct() {
        struct Juju;

        match thread::spawn(move|| {
            panic!(Juju)
        }).join() {
            Err(ref e) if e.is::<Juju>() => {}
            Err(_) | Ok(()) => panic!()
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
                thread::sleep(Duration::from_millis(50));
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
    fn test_thread_id_equal() {
        assert!(thread::current().id() == thread::current().id());
    }

    #[test]
    fn test_thread_id_not_equal() {
        let spawned_id = thread::spawn(|| thread::current().id()).join().unwrap();
        assert!(thread::current().id() != spawned_id);
    }

    // NOTE: the corresponding test for stderr is in run-pass/thread-stderr, due
    // to the test harness apparently interfering with stderr configuration.
}

#[cfg(not(target_os = "emscripten"))]
mod local {
    use sync::mpsc::{channel, Sender};
    use cell::{Cell, UnsafeCell};
    use thread;

    struct Foo(Sender<()>);

    impl Drop for Foo {
        fn drop(&mut self) {
            let Foo(ref s) = *self;
            s.send(()).unwrap();
        }
    }

    #[test]
    fn smoke_no_dtor() {
        thread_local!(static FOO: Cell<i32> = Cell::new(1));

        FOO.with(|f| {
            assert_eq!(f.get(), 1);
            f.set(2);
        });
        let (tx, rx) = channel();
        let _t = thread::spawn(move|| {
            FOO.with(|f| {
                assert_eq!(f.get(), 1);
            });
            tx.send(()).unwrap();
        });
        rx.recv().unwrap();

        FOO.with(|f| {
            assert_eq!(f.get(), 2);
        });
    }

    #[test]
    fn states() {
        struct Foo;
        impl Drop for Foo {
            fn drop(&mut self) {
                assert!(FOO.try_with(|_| ()).is_err());
            }
        }
        thread_local!(static FOO: Foo = Foo);

        thread::spawn(|| {
            assert!(FOO.try_with(|_| ()).is_ok());
        }).join().ok().unwrap();
    }

    #[test]
    fn smoke_dtor() {
        thread_local!(static FOO: UnsafeCell<Option<Foo>> = UnsafeCell::new(None));

        let (tx, rx) = channel();
        let _t = thread::spawn(move|| unsafe {
            let mut tx = Some(tx);
            FOO.with(|f| {
                *f.get() = Some(Foo(tx.take().unwrap()));
            });
        });
        rx.recv().unwrap();
    }

    #[test]
    fn circular() {
        struct S1;
        struct S2;
        thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell::new(None));
        thread_local!(static K2: UnsafeCell<Option<S2>> = UnsafeCell::new(None));
        static mut HITS: u32 = 0;

        impl Drop for S1 {
            fn drop(&mut self) {
                unsafe {
                    HITS += 1;
                    if K2.try_with(|_| ()).is_err() {
                        assert_eq!(HITS, 3);
                    } else {
                        if HITS == 1 {
                            K2.with(|s| *s.get() = Some(S2));
                        } else {
                            assert_eq!(HITS, 3);
                        }
                    }
                }
            }
        }
        impl Drop for S2 {
            fn drop(&mut self) {
                unsafe {
                    HITS += 1;
                    assert!(K1.try_with(|_| ()).is_ok());
                    assert_eq!(HITS, 2);
                    K1.with(|s| *s.get() = Some(S1));
                }
            }
        }

        thread::spawn(move|| {
            drop(S1);
        }).join().ok().unwrap();
    }

    #[test]
    fn self_referential() {
        struct S1;
        thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell::new(None));

        impl Drop for S1 {
            fn drop(&mut self) {
                assert!(K1.try_with(|_| ()).is_err());
            }
        }

        thread::spawn(move|| unsafe {
            K1.with(|s| *s.get() = Some(S1));
        }).join().ok().unwrap();
    }

    // Note that this test will deadlock if TLS destructors aren't run (this
    // requires the destructor to be run to pass the test).
    #[test]
    fn dtors_in_dtors_in_dtors() {
        struct S1(Sender<()>);
        thread_local!(static K1: UnsafeCell<Option<S1>> = UnsafeCell::new(None));
        thread_local!(static K2: UnsafeCell<Option<Foo>> = UnsafeCell::new(None));

        impl Drop for S1 {
            fn drop(&mut self) {
                let S1(ref tx) = *self;
                unsafe {
                    let _ = K2.try_with(|s| *s.get() = Some(Foo(tx.clone())));
                }
            }
        }

        let (tx, rx) = channel();
        let _t = thread::spawn(move|| unsafe {
            let mut tx = Some(tx);
            K1.with(|s| *s.get() = Some(S1(tx.take().unwrap())));
        });
        rx.recv().unwrap();
    }
}

mod local_dynamic {
    use cell::RefCell;
    use collections::HashMap;

    #[test]
    fn smoke() {
        fn square(i: i32) -> i32 { i * i }
        thread_local!(static FOO: i32 = square(3));

        FOO.with(|f| {
            assert_eq!(*f, 9);
        });
    }

    #[test]
    fn hashmap() {
        fn map() -> RefCell<HashMap<i32, i32>> {
            let mut m = HashMap::new();
            m.insert(1, 2);
            RefCell::new(m)
        }
        thread_local!(static FOO: RefCell<HashMap<i32, i32>> = map());

        FOO.with(|map| {
            assert_eq!(map.borrow()[&1], 2);
        });
    }

    #[test]
    fn refcell_vec() {
        thread_local!(static FOO: RefCell<Vec<u32>> = RefCell::new(vec![1, 2, 3]));

        FOO.with(|vec| {
            assert_eq!(vec.borrow().len(), 3);
            vec.borrow_mut().push(4);
            assert_eq!(vec.borrow()[3], 4);
        });
    }
}
