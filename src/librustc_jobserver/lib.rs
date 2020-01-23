//! rustc wants to manage its jobserver pool such that it never keeps a token
//! around for too long if it's not being used (i.e., eagerly return tokens), so
//! that Cargo can spawn more rustcs to go around.
//!
//! rustc also has a process-global implicit token when it starts, which we keep
//! track of -- we cannot release it to Cargo, and we want to make sure that if
//! it is released we *must* unblock a thread of execution onto it (otherwise we
//! will deadlock on it almost for sure).
//!
//! So, when we start up, we have an implicit token and no acquired tokens from
//! Cargo.
//!
//! We immediately on startup spawn a thread into the background to manage
//! communication with the jobserver (otherwise it's too hard to work with the
//! jobserver API). This is non-ideal, and it would be good to avoid, but
//! currently that cost is pretty much required for correct functionality, as we
//! must be able to atomically wait on both a Condvar (for other threads
//! releasing the implicit token) and the jobserver itself. That's not possible
//! with the jobserver API today unless we spawn up an additional thread.
//!
//! There are 3 primary APIs this crate exposes:
//!  * acquire()
//!  * release()
//!  * acquire_from_request()
//!  * request_token()
//!
//! The first two, acquire and release, are blocking functions which acquire
//! and release a jobserver token.
//!
//! The latter two help manage async requesting of tokens: specifically,
//! acquire_from_request() will block on acquiring token but will not request it
//! from the jobserver itself, whereas the last one just requests a token (and
//! should return pretty quickly, i.e., it does not block on some event).
//!
//! -------------------------------------
//!
//! We also have two modes to manage here. In the primary (default) mode we
//! communicate directly with the underlying jobserver (i.e., all
//! acquire/release requests fall through to the jobserver crate's
//! acquire/release functions).
//!
//! This can be quite poor for scalability, as at least on current Linux
//! kernels, each release on the jobserver will trigger the kernel to wake up
//! *all* waiters instead of just one, which, if you have lots of threads
//! waiting, is quite bad.
//!
//! For that reason, we have a second mode which utilizes Cargo to improve
//! scaling here. In that mode, we have slightly altered communication with the
//! jobserver. Instead of just blocking on the jobserver, we will instead first
//! print to stderr a JSON message indicating that we're interested in receiving
//! a jobserver token, and only then block on actually receiving said token. On
//! release, we don't write into the jobserver at all, instead merely printing
//! out that we've released a token.
//!
//! Note that the second mode allows Cargo to hook up each rustc with its own
//! jobserver (i.e., one per rustc process) and then fairly easily make sure to
//! fulfill the requests from rustc and such. Ultimately, that means that we
//! have just one rustc thread waiting on the jobserver: a solution that is
//! nearly optimal for scalability.

use jobserver::Client;
use lazy_static::lazy_static;
use rustc_serialize::json::as_json;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

lazy_static! {
    // We can only call `from_env` once per process

    // Note that this is unsafe because it may misinterpret file descriptors
    // on Unix as jobserver file descriptors. We hopefully execute this near
    // the beginning of the process though to ensure we don't get false
    // positives, or in other words we try to execute this before we open
    // any file descriptors ourselves.
    //
    // Pick a "reasonable maximum" if we don't otherwise have
    // a jobserver in our environment, capping out at 32 so we
    // don't take everything down by hogging the process run queue.
    //
    // Also note that we stick this in a global because there could be
    // multiple rustc instances in this process, and the jobserver is
    // per-process.
    static ref GLOBAL_CLIENT: Client = unsafe {
        Client::from_env().unwrap_or_else(|| {
            log::trace!("initializing fresh jobserver (not from env)");
            let client = Client::new(32).expect("failed to create jobserver");
            // Acquire a token for the main thread which we can release later
            client.acquire_raw().ok();
            client
        })
    };

    static ref HELPER: Mutex<Helper> = Mutex::new(Helper::new());
}

// These are the values for TOKEN_REQUESTS, which is an enum between these
// different options.
//
//
// It takes the following values:
//  * EMPTY: not yet set
//  * CARGO_REQUESTED: we're in the jobserver-per-rustc mode
//  * MAKE_REQUESTED: legacy global jobserver client
const EMPTY: usize = 0;
const CARGO_REQUESTED: usize = 1;
const MAKE_REQUESTED: usize = 2;
static TOKEN_REQUESTS: AtomicUsize = AtomicUsize::new(EMPTY);

fn should_notify() -> bool {
    let value = TOKEN_REQUESTS.load(Ordering::SeqCst);
    assert!(value != EMPTY, "jobserver must be initialized");
    value == CARGO_REQUESTED
}

/// This will only adjust the global value to the new value of token_requests
/// the first time it's called, which means that you want to make sure that the
/// first call you make has the right value for `token_requests`. We try to help
/// out a bit by making sure that this is called before any interaction with the
/// jobserver (which usually happens almost immediately as soon as rustc does
/// anything due to spawning up the Rayon threadpool).
///
/// Unfortunately the jobserver is inherently a global resource (we can't
/// have more than one) so the token requesting strategy must likewise be global.
///
/// Usually this doesn't matter too much, as you're not wanting to set the token
/// requests unless you're in the one-rustc-per-process model, and we help out
/// here a bit by not resetting it once it's set (i.e., only the first init will
/// change the value).
pub fn initialize(token_requests: bool) {
    lazy_static::initialize(&GLOBAL_CLIENT);
    lazy_static::initialize(&HELPER);
    let previous = TOKEN_REQUESTS.compare_and_swap(
        EMPTY,
        if token_requests { CARGO_REQUESTED } else { MAKE_REQUESTED },
        Ordering::SeqCst,
    );
    if previous == EMPTY {
        log::info!("initialized rustc jobserver, set token_requests={:?}", token_requests);
    }
}

pub struct Helper {
    helper: jobserver::HelperThread,
    tokens: usize,
    requests: Arc<Mutex<VecDeque<Box<dyn FnOnce(Acquired) + Send>>>>,
}

impl Helper {
    fn new() -> Self {
        let requests: Arc<Mutex<VecDeque<Box<dyn FnOnce(Acquired) + Send>>>> =
            Arc::new(Mutex::new(VecDeque::new()));
        let requests2 = requests.clone();
        let helper = GLOBAL_CLIENT
            .clone()
            .into_helper_thread(move |token| {
                log::trace!("Helper thread token sending into channel");
                // We've acquired a token, but we need to not use it as we have our own
                // custom release-on-drop struct since we'll want different logic than
                // just normally releasing the token in this case.
                //
                // On unix this unfortunately means that we lose the specific byte that
                // was in the pipe (i.e., we just write back the same byte all the time)
                // but that's not expected to be a problem.
                token.expect("acquire token").drop_without_releasing();
                if let Some(sender) = requests2.lock().unwrap().pop_front() {
                    sender(Acquired::new());
                }
            })
            .expect("spawned helper");
        Helper { helper, tokens: 1, requests }
    }

    // This blocks on acquiring a token (that must have been previously
    // requested).
    fn acquire_token_from_prior_request(&mut self) -> Acquired {
        if self.tokens == 0 {
            self.tokens += 1;
            return Acquired::new();
        }

        let receiver = Arc::new((Mutex::new(None), Condvar::new()));
        let receiver2 = receiver.clone();

        self.requests.lock().unwrap().push_back(Box::new(move |token| {
            let mut slot = receiver.0.lock().unwrap();
            *slot = Some(token);
            receiver.1.notify_one();
        }));

        let (lock, cvar) = &*receiver2;
        let mut guard = cvar.wait_while(lock.lock().unwrap(), |slot| slot.is_none()).unwrap();

        self.tokens += 1;
        guard.take().unwrap()
    }

    fn release_token(&mut self) {
        let mut requests = self.requests.lock().unwrap();

        self.tokens -= 1;

        if self.tokens == 0 {
            // If there is a sender, then it needs to be given this token.
            if let Some(sender) = requests.pop_front() {
                sender(Acquired::new());
                return;
            }

            return;
        }

        if should_notify() {
            eprintln!("{}", as_json(&JobserverNotification { jobserver_event: Event::Release }));
        } else {
            GLOBAL_CLIENT.release_raw().unwrap();
        }
    }

    pub fn request_token(&self) {
        log::trace!("{:?} requesting token", std::thread::current().id());
        // Just notify, don't actually acquire here.
        notify_acquiring_token();
        self.helper.request_token();
    }
}

#[must_use]
pub struct Acquired {
    armed: bool,
}

impl Drop for Acquired {
    fn drop(&mut self) {
        if self.armed {
            release_thread();
        }
    }
}

impl Acquired {
    fn new() -> Self {
        Self { armed: true }
    }

    fn disarm(mut self) {
        self.armed = false;
    }
}

#[derive(RustcEncodable)]
enum Event {
    WillAcquire,
    Release,
}

#[derive(RustcEncodable)]
struct JobserverNotification {
    jobserver_event: Event,
}

// Unlike releasing tokens, there's not really a "one size fits all" approach, as we have two
// primary ways of acquiring a token: via the helper thread, and via the acquire_thread function.
fn notify_acquiring_token() {
    if should_notify() {
        eprintln!("{}", as_json(&JobserverNotification { jobserver_event: Event::WillAcquire }));
    }
}

pub fn request_token(f: impl FnOnce(Acquired) + Send + 'static) {
    HELPER.lock().unwrap().requests.lock().unwrap().push_back(Box::new(move |token| {
        f(token);
    }));
}

pub fn acquire_from_request() -> Acquired {
    HELPER.lock().unwrap().acquire_token_from_prior_request()
}

pub fn acquire_thread() {
    HELPER.lock().unwrap().request_token();
    HELPER.lock().unwrap().acquire_token_from_prior_request().disarm();
}

pub fn release_thread() {
    HELPER.lock().unwrap().release_token();
}
