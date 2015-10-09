use cell::RefCell;
use thread::Thread;
use thread::LocalKeyState;

thread_local! { static CURRENT_THREAD: RefCell<Option<Thread>> = RefCell::new(None) }

fn with<R, F>(f: F) -> Option<R> where F: FnOnce(&mut Thread) -> R {
    if CURRENT_THREAD.state() == LocalKeyState::Destroyed {
        return None
    }

    CURRENT_THREAD.with(move |c| {
        let mut c = c.borrow_mut();
        if c.is_none() {
            *c = Some(Thread::new(None))
        }
        c.as_mut().map(f)
    })
}

pub fn current_thread() -> Option<Thread> {
    with(|thread| thread.clone())
}

pub fn set_current_thread(t: Thread) {
    CURRENT_THREAD.with(move |c| *c.borrow_mut() = Some(t));
}
