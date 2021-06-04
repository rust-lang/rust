// run-pass
// no-prefer-dynamic
// ignore-wasm32-bare no libc
// ignore-windows
// ignore-sgx no libc
// ignore-emscripten no processes
// ignore-sgx no processes
// ignore-android: FIXME(#85261)

#![feature(bench_black_box)]
#![feature(rustc_private)]
#![feature(never_type)]
#![feature(panic_always_abort)]

extern crate libc;

use std::alloc::{GlobalAlloc, Layout};
use std::fmt;
use std::panic::{self, panic_any};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::{self, Command, ExitStatus};
use std::sync::atomic::{AtomicU32, Ordering};

use libc::c_int;

/// This stunt allocator allows us to spot heap allocations in the child.
struct PidChecking<A> {
    parent: A,
    require_pid: AtomicU32,
}

#[global_allocator]
static ALLOCATOR: PidChecking<std::alloc::System> = PidChecking {
    parent: std::alloc::System,
    require_pid: AtomicU32::new(0),
};

impl<A> PidChecking<A> {
    fn engage(&self) {
        let parent_pid = process::id();
        eprintln!("engaging allocator trap, parent pid={}", parent_pid);
        self.require_pid.store(parent_pid, Ordering::Release);
    }
    fn check(&self) {
        let require_pid = self.require_pid.load(Ordering::Acquire);
        if require_pid != 0 {
            let actual_pid = process::id();
            if require_pid != actual_pid {
                unsafe {
                    libc::raise(libc::SIGUSR1);
                }
            }
        }
    }
}

unsafe impl<A:GlobalAlloc> GlobalAlloc for PidChecking<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.check();
        self.parent.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.check();
        self.parent.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.check();
        self.parent.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.check();
        self.parent.realloc(ptr, layout, new_size)
    }
}

fn expect_aborted(status: ExitStatus) {
    dbg!(status);
    let signal = status.signal().expect("expected child process to die of signal");
    assert!(signal == libc::SIGABRT || signal == libc::SIGILL || signal == libc::SIGTRAP);
}

fn main() {
    ALLOCATOR.engage();

    fn run(do_panic: &dyn Fn()) -> ExitStatus {
        let child = unsafe { libc::fork() };
        assert!(child >= 0);
        if child == 0 {
            panic::always_abort();
            do_panic();
            process::exit(0);
        }
        let mut status: c_int = 0;
        let got = unsafe { libc::waitpid(child, &mut status, 0) };
        assert_eq!(got, child);
        let status = ExitStatus::from_raw(status.into());
        status
    }

    fn one(do_panic: &dyn Fn()) {
        let status = run(do_panic);
        expect_aborted(status);
    }

    one(&|| panic!());
    one(&|| panic!("some message"));
    one(&|| panic!("message with argument: {}", 42));

    #[derive(Debug)]
    struct Wotsit { }
    one(&|| panic_any(Wotsit { }));

    let mut c = Command::new("echo");
    unsafe {
        c.pre_exec(|| panic!("{}", "crash now!"));
    }
    let st = c.status().expect("failed to get command status");
    expect_aborted(st);

    struct DisplayWithHeap;
    impl fmt::Display for DisplayWithHeap {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
            let s = vec![0; 100];
            let s = std::hint::black_box(s);
            write!(f, "{:?}", s)
        }
    }

    // Some panics in the stdlib that we want not to allocate, as
    // otherwise these facilities become impossible to use in the
    // child after fork, which is really quite awkward.

    one(&|| { None::<DisplayWithHeap>.unwrap(); });
    one(&|| { None::<DisplayWithHeap>.expect("unwrapped a none"); });
    one(&|| { std::str::from_utf8(b"\xff").unwrap(); });
    one(&|| {
        let x = [0, 1, 2, 3];
        let y = x[std::hint::black_box(4)];
        let _z = std::hint::black_box(y);
    });

    // Finally, check that our stunt allocator can actually catch an allocation after fork.
    // ie, that our test is effective.

    let status = run(&|| panic!("allocating to display... {}", DisplayWithHeap));
    dbg!(status);
    assert_eq!(status.signal(), Some(libc::SIGUSR1));
}
