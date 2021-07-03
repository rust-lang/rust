use super::*;
use crate::panic::{RefUnwindSafe, UnwindSafe};
use crate::sync::mpsc::sync_channel;
use crate::thread;

#[test]
fn stdout_unwind_safe() {
    assert_unwind_safe::<Stdout>();
}
#[test]
fn stdoutlock_unwind_safe() {
    assert_unwind_safe::<StdoutLock<'_>>();
    assert_unwind_safe::<StdoutLock<'static>>();
}
#[test]
fn stderr_unwind_safe() {
    assert_unwind_safe::<Stderr>();
}
#[test]
fn stderrlock_unwind_safe() {
    assert_unwind_safe::<StderrLock<'_>>();
    assert_unwind_safe::<StderrLock<'static>>();
}

fn assert_unwind_safe<T: UnwindSafe + RefUnwindSafe>() {}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn panic_doesnt_poison() {
    thread::spawn(|| {
        let _a = stdin();
        let _a = _a.lock();
        let _a = stdout();
        let _a = _a.lock();
        let _a = stderr();
        let _a = _a.lock();
        panic!();
    })
    .join()
    .unwrap_err();

    let _a = stdin();
    let _a = _a.lock();
    let _a = stdout();
    let _a = _a.lock();
    let _a = stderr();
    let _a = _a.lock();
}

#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn test_lock_stderr() {
    test_lock(stderr, stderr_locked);
}
#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn test_lock_stdin() {
    test_lock(stdin, stdin_locked);
}
#[test]
#[cfg_attr(target_os = "emscripten", ignore)]
fn test_lock_stdout() {
    test_lock(stdout, stdout_locked);
}

// Helper trait to make lock testing function generic.
trait Stdio<'a>: 'static
where
    Self::Lock: 'a,
{
    type Lock;
    fn lock(&'a self) -> Self::Lock;
}
impl<'a> Stdio<'a> for Stderr {
    type Lock = StderrLock<'a>;
    fn lock(&'a self) -> StderrLock<'a> {
        self.lock()
    }
}
impl<'a> Stdio<'a> for Stdin {
    type Lock = StdinLock<'a>;
    fn lock(&'a self) -> StdinLock<'a> {
        self.lock()
    }
}
impl<'a> Stdio<'a> for Stdout {
    type Lock = StdoutLock<'a>;
    fn lock(&'a self) -> StdoutLock<'a> {
        self.lock()
    }
}

// Helper trait to make lock testing function generic.
trait StdioOwnedLock: 'static {}
impl StdioOwnedLock for StderrLock<'static> {}
impl StdioOwnedLock for StdinLock<'static> {}
impl StdioOwnedLock for StdoutLock<'static> {}

// Tests locking on stdio handles by starting two threads and checking that
// they block each other appropriately.
fn test_lock<T, U>(get_handle: fn() -> T, get_locked: fn() -> U)
where
    T: for<'a> Stdio<'a>,
    U: StdioOwnedLock,
{
    // State enum to track different phases of the test, primarily when
    // each lock is acquired and released.
    #[derive(Debug, PartialEq)]
    enum State {
        Start1,
        Acquire1,
        Start2,
        Release1,
        Acquire2,
        Release2,
    }
    use State::*;
    // Logging vector to be checked to make sure lock acquisitions and
    // releases happened in the correct order.
    let log = Arc::new(Mutex::new(Vec::new()));
    let ((tx1, rx1), (tx2, rx2)) = (sync_channel(0), sync_channel(0));
    let th1 = {
        let (log, tx) = (Arc::clone(&log), tx1);
        thread::spawn(move || {
            log.lock().unwrap().push(Start1);
            let handle = get_handle();
            {
                let locked = handle.lock();
                log.lock().unwrap().push(Acquire1);
                tx.send(Acquire1).unwrap(); // notify of acquisition
                tx.send(Release1).unwrap(); // wait for release command
                log.lock().unwrap().push(Release1);
            }
            tx.send(Acquire1).unwrap(); // wait for th2 acquire
            {
                let locked = handle.lock();
                log.lock().unwrap().push(Acquire1);
            }
            log.lock().unwrap().push(Release1);
        })
    };
    let th2 = {
        let (log, tx) = (Arc::clone(&log), tx2);
        thread::spawn(move || {
            tx.send(Start2).unwrap(); // wait for start command
            let locked = get_locked();
            log.lock().unwrap().push(Acquire2);
            tx.send(Acquire2).unwrap(); // notify of acquisition
            tx.send(Release2).unwrap(); // wait for release command
            log.lock().unwrap().push(Release2);
        })
    };
    assert_eq!(rx1.recv().unwrap(), Acquire1); // wait for th1 acquire
    log.lock().unwrap().push(Start2);
    assert_eq!(rx2.recv().unwrap(), Start2); // block th2
    assert_eq!(rx1.recv().unwrap(), Release1); // release th1
    assert_eq!(rx2.recv().unwrap(), Acquire2); // wait for th2 acquire
    assert_eq!(rx1.recv().unwrap(), Acquire1); // block th1
    assert_eq!(rx2.recv().unwrap(), Release2); // release th2
    th2.join().unwrap();
    th1.join().unwrap();
    assert_eq!(
        *log.lock().unwrap(),
        [Start1, Acquire1, Start2, Release1, Acquire2, Release2, Acquire1, Release1]
    );
}
