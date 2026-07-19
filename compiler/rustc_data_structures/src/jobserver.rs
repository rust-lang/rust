use std::sync::{Arc, LazyLock, OnceLock};

pub use jobserver_crate::Acquired;
use jobserver_crate::{Client, FromEnv, FromEnvErrorKind, HelperThread};
use parking_lot::{Condvar, Mutex};

// We stick the jobserver client into a global and initialize it once, because there could be
// multiple compiler instances in this process, and the jobserver is per-process.
static GLOBAL_CLIENT: LazyLock<Result<Client, String>> = LazyLock::new(|| {
    // Safety: the checked client construction ensures that the jobserver file descriptors
    // (if any) are open and valid. We also try to initialize the jobserver as early as possible
    // to avoid unrelated file descriptors with matching values becoming open and valid between
    // the process start and the jobserver initialization.
    let FromEnv { client, var } = unsafe { Client::from_env_ext(true) };

    let error = match client {
        Ok(client) => return Ok(client),
        Err(e) => e,
    };

    if matches!(
        error.kind(),
        FromEnvErrorKind::NoEnvVar
            | FromEnvErrorKind::NoJobserver
            | FromEnvErrorKind::NegativeFd
            | FromEnvErrorKind::Unsupported
    ) {
        return Ok(default_client());
    }

    // Environment specifies jobserver, but it looks incorrect.
    // Safety: `error.kind()` should be `NoEnvVar` if `var == None`.
    let (name, value) = var.unwrap();
    Err(format!(
        "failed to connect to jobserver from environment variable `{name}={:?}`: {error}",
        value
    ))
});

// Creates a new jobserver if there's no inherited one.
fn default_client() -> Client {
    // Pick a "reasonable maximum" capping out at 32
    // so we don't take everything down by hogging the process run queue.
    // The fixed number is used to have deterministic compilation across machines.
    let client = Client::new(32).expect("failed to create jobserver");

    // Acquire the single token that is always held by the rustc process.
    // This is an equivalent of the single token held by a higher level build tool while running
    // this instance of rustc. This token is never released - if we are here, then rustc owns the
    // jobserver, it is teared down when rustc exits, and there's no one to return the token to.
    client.acquire_raw().ok();

    client
}

static GLOBAL_CLIENT_CHECKED: OnceLock<Client> = OnceLock::new();

/// Initializes a jobserver client for the current rustc process.
/// If inheriting jobserver from the environment fails for some reason, an new jobserver owned by
/// the current rustc process will be created. If the inheritance failure reason is non-benign,
/// the passed callback will be used to report the error.
pub fn initialize_checked(report: impl FnOnce(&'static str)) {
    let client_checked = match &*GLOBAL_CLIENT {
        Ok(client) => client.clone(),
        Err(e) => {
            report(e);
            default_client()
        }
    };
    GLOBAL_CLIENT_CHECKED.set(client_checked).ok();
}

/// Returns the jobserver client previously initialized by `initialize_checked`.
///
/// # Assumptions about holding jobserver tokens
///
/// Rustc process must always hold a single token to avoid being permanently starved and blocked.
/// - If the jobserver is inherited from a higher level build tool, the assumption is that the tool
///   will hold the token and not release it until the rustc process exits.
/// - If the jobserver is owned by the current rustc, the token is acquired by `default_client`.
///
/// To avoid releasing the last token, users of the client returned by this function must ensure
/// that they never release more tokens than was previously explicitly acquired.
/// Example of a sequence that can accidentally release the last token:
/// `release_raw` -> `wait` -> `acquire_raw`.
/// To avoid situations like this use the `jobserver::Proxy` wrapper instead,
/// it will ensure that the last token is never released.
pub fn client() -> Client {
    GLOBAL_CLIENT_CHECKED.get().expect("uninitialized jobserver client").clone()
}

struct ProxyData {
    /// The number of tokens assigned to actively working threads,
    /// possibly including the single permanently held token.
    /// If this number is 0, the single token is still held by the process,
    /// but is not currently used for active CPU work.
    /// This can happen, for example, if the main thread is waiting for something,
    /// in that case some other thread can start using this token to do work.
    used: u16,
    /// The number of threads currently requesting a token and waiting.
    /// If the proxy releases a token it can immediately give it to one of these threads
    /// without going through the real jobserver.
    pending: u16,
}

/// A wrapper around jobserver client used for two purposes:
/// - Ensuring that the single token that must be permanently held by the rustc process
///   cannot be accidentally released.
/// - "Token buffering", immediately acquiring freshly released tokens if necessary,
///   without going through the real jobserver.
pub struct Proxy {
    /// The wrapped jobserver client.
    client: Client,
    /// Helper thread associated with the wrapped client.
    helper: OnceLock<HelperThread>,
    /// The proxy's own data.
    data: Mutex<ProxyData>,
    /// Threads which are currently waiting for a token will wait on this condvar.
    wake_pending: Condvar,
}

impl Proxy {
    pub fn new() -> Arc<Self> {
        let proxy = Arc::new(Proxy {
            client: client(),
            // Assume that the main thread is actively doing work when it creates the proxy.
            data: Mutex::new(ProxyData { used: 1, pending: 0 }),
            wake_pending: Condvar::new(),
            helper: OnceLock::new(),
        });
        let proxy_ = Arc::clone(&proxy);
        let helper = proxy
            .client
            .clone()
            .into_helper_thread(move |token| {
                // Reminder: this callback runs when the helper acquires a token.
                if let Ok(token) = token {
                    let mut data = proxy_.data.lock();
                    if data.pending > 0 {
                        // The token is still needed, give it to one of the waiting threads.
                        token.drop_without_releasing();
                        assert!(data.used > 0);
                        data.used += 1;
                        data.pending -= 1;
                        proxy_.wake_pending.notify_one();
                    } else {
                        // The token is no longer needed, release it by dropping.
                        drop(data);
                        drop(token);
                    }
                }
            })
            .expect("failed to spawn helper thread");
        proxy.helper.set(helper).unwrap();
        proxy
    }

    /// Acquires a token, possibly using some buffered tokens as an optimization.
    /// May block and wait until the token is available.
    pub fn acquire_thread(&self) {
        let mut data = self.data.lock();

        if data.used == 0 {
            // No threads are doing any active work, but we are still holding the last token.
            // Give that token to the current thread.
            assert_eq!(data.pending, 0);
            data.used += 1;
        } else {
            // Request a token from the helper thread, this is a non-blocking operation.
            // Then wait until this or some other request succeeds.
            self.helper.get().unwrap().request_token();
            data.pending += 1;
            self.wake_pending.wait(&mut data);
        }
    }

    /// Releases a token, possibly immediately giving it to some other thread as an optimization.
    /// Makes sure that the last token is never actually released to the wrapped jobserver.
    pub fn release_thread(&self) {
        let mut data = self.data.lock();

        if data.pending > 0 {
            // Immediately give the released token to one of the waiting threads.
            data.pending -= 1;
            self.wake_pending.notify_one();
        } else {
            data.used -= 1;

            // Release the token to the wrapped jobserver, unless it's the last one in the process.
            if data.used > 0 {
                drop(data);
                self.client.release_raw().ok();
            }
        }
    }
}
