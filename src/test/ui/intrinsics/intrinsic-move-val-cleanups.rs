// run-pass
#![allow(unused_unsafe)]
#![allow(unreachable_code)]
// ignore-emscripten no threads support
#![allow(stable_features)]

// This test is checking that the move_val_init intrinsic is
// respecting cleanups for both of its argument expressions.
//
// In other words, if either DEST or SOURCE in
//
//   `intrinsics::move_val_init(DEST, SOURCE)
//
// introduce temporaries that require cleanup, and SOURCE panics, then
// make sure the cleanups still occur.

#![feature(core_intrinsics, sync_poison)]

use std::cell::RefCell;
use std::intrinsics;
use std::sync::{Arc, LockResult, Mutex, MutexGuard};
use std::thread;

type LogEntry = (&'static str, i32);
type Guarded = RefCell<Vec<LogEntry>>;
#[derive(Clone)]
struct Log(Arc<Mutex<Guarded>>);
struct Acquired<'a>(MutexGuard<'a, Guarded>);
type LogState = (MutexWas, &'static [LogEntry]);

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum MutexWas { Poisoned, NotPoisoned }

impl Log {
    fn lock(&self) -> LockResult<MutexGuard<RefCell<Vec<LogEntry>>>> { self.0.lock() }
    fn acquire(&self) -> Acquired { Acquired(self.0.lock().unwrap()) }
}

impl<'a> Acquired<'a> {
    fn log(&self, s: &'static str, i: i32) { self.0.borrow_mut().push((s, i)); }
}

const TEST1_EXPECT: LogState = (MutexWas::NotPoisoned,
                                &[("double-check non-poisoning path", 1)
                                  ]);

fn test1(log: Log) {
    {
        let acq = log.acquire();
        acq.log("double-check non-poisoning path", 1);
    }
    panic!("every test ends in a panic");
}

const TEST2_EXPECT: LogState = (MutexWas::Poisoned,
                                &[("double-check poisoning path", 1),
                                  ("and multiple log entries", 2),
                                  ]);
fn test2(log: Log) {
    let acq = log.acquire();
    acq.log("double-check poisoning path", 1);
    acq.log("and multiple log entries", 2);
    panic!("every test ends in a panic");
}

struct LogOnDrop<'a>(&'a Acquired<'a>, &'static str, i32);
impl<'a> Drop for LogOnDrop<'a> {
    fn drop(&mut self) {
        self.0.log(self.1, self.2);
    }
}

const TEST3_EXPECT: LogState = (MutexWas::Poisoned,
                                &[("double-check destructors can log", 1),
                                  ("drop d2", 2),
                                  ("drop d1", 3),
                                  ]);
fn test3(log: Log) {
    let acq = log.acquire();
    acq.log("double-check destructors can log", 1);
    let _d1 = LogOnDrop(&acq, "drop d1", 3);
    let _d2 = LogOnDrop(&acq, "drop d2", 2);
    panic!("every test ends in a panic");
}

// The *real* tests of panic-handling for move_val_init intrinsic
// start here.

const TEST4_EXPECT: LogState = (MutexWas::Poisoned,
                                &[("neither arg panics", 1),
                                  ("drop temp LOD", 2),
                                  ("drop temp LOD", 3),
                                  ("drop dest_b", 4),
                                  ("drop dest_a", 5),
                                  ]);
fn test4(log: Log) {
    let acq = log.acquire();
    acq.log("neither arg panics", 1);
    let mut dest_a = LogOnDrop(&acq, "a will be overwritten, not dropped", 0);
    let mut dest_b = LogOnDrop(&acq, "b will be overwritten, not dropped", 0);
    unsafe {
        intrinsics::move_val_init({ LogOnDrop(&acq, "drop temp LOD", 2); &mut dest_a },
                                  LogOnDrop(&acq, "drop dest_a", 5));
        intrinsics::move_val_init(&mut dest_b, { LogOnDrop(&acq, "drop temp LOD", 3);
                                                 LogOnDrop(&acq, "drop dest_b", 4) });
    }
    panic!("every test ends in a panic");
}


// Check that move_val_init(PANIC, SOURCE_EXPR) never evaluates SOURCE_EXPR
const TEST5_EXPECT: LogState = (MutexWas::Poisoned,
                                &[("first arg panics", 1),
                                  ("drop orig dest_a", 2),
                                  ]);
fn test5(log: Log) {
    let acq = log.acquire();
    acq.log("first arg panics", 1);
    let mut _dest_a = LogOnDrop(&acq, "drop orig dest_a", 2);
    unsafe {
        intrinsics::move_val_init({ panic!("every test ends in a panic") },
                                  LogOnDrop(&acq, "we never get here", 0));
    }
}

// Check that move_val_init(DEST_EXPR, PANIC) cleans up temps from DEST_EXPR.
const TEST6_EXPECT: LogState = (MutexWas::Poisoned,
                                &[("second arg panics", 1),
                                  ("drop temp LOD", 2),
                                  ("drop orig dest_a", 3),
                                  ]);
fn test6(log: Log) {
    let acq = log.acquire();
    acq.log("second arg panics", 1);
    let mut dest_a = LogOnDrop(&acq, "drop orig dest_a", 3);
    unsafe {
        intrinsics::move_val_init({ LogOnDrop(&acq, "drop temp LOD", 2); &mut dest_a },
                                  { panic!("every test ends in a panic"); });
    }
}

// Check that move_val_init(DEST_EXPR, COMPLEX_PANIC) cleans up temps from COMPLEX_PANIC.
const TEST7_EXPECT: LogState = (MutexWas::Poisoned,
                                &[("second arg panics", 1),
                                  ("drop temp LOD", 2),
                                  ("drop temp LOD", 3),
                                  ("drop orig dest_a", 4),
                                  ]);
fn test7(log: Log) {
    let acq = log.acquire();
    acq.log("second arg panics", 1);
    let mut dest_a = LogOnDrop(&acq, "drop orig dest_a", 4);
    unsafe {
        intrinsics::move_val_init({ LogOnDrop(&acq, "drop temp LOD", 2); &mut dest_a },
                                  { LogOnDrop(&acq, "drop temp LOD", 3);
                                    panic!("every test ends in a panic"); });
    }
}

const TEST_SUITE: &'static [(&'static str, fn (Log), LogState)] =
    &[("test1", test1, TEST1_EXPECT),
      ("test2", test2, TEST2_EXPECT),
      ("test3", test3, TEST3_EXPECT),
      ("test4", test4, TEST4_EXPECT),
      ("test5", test5, TEST5_EXPECT),
      ("test6", test6, TEST6_EXPECT),
      ("test7", test7, TEST7_EXPECT),
      ];

fn main() {
    for &(name, test, expect) in TEST_SUITE {
        let log = Log(Arc::new(Mutex::new(RefCell::new(Vec::new()))));
        let ret = { let log = log.clone(); thread::spawn(move || test(log)).join() };
        assert!(ret.is_err(), "{} must end with panic", name);
        {
            let l = log.lock();
            match l {
                Ok(acq) => {
                    assert_eq!((MutexWas::NotPoisoned, &acq.borrow()[..]), expect);
                    println!("{} (unpoisoned) log: {:?}", name, *acq);
                }
                Err(e) => {
                    let acq = e.into_inner();
                    assert_eq!((MutexWas::Poisoned, &acq.borrow()[..]), expect);
                    println!("{} (poisoned) log: {:?}", name, *acq);
                }
            }
        }
    }
}
