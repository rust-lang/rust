use std::sync::{Arc, LazyLock, OnceLock};

pub use jobserver_crate::{Acquired, Client, HelperThread};
use jobserver_crate::{FromEnv, FromEnvErrorKind};
use parking_lot::{Condvar, Mutex};

// We can only call `from_env_ext` once per process

// We stick this in a global because there could be multiple rustc instances
// in this process, and the jobserver is per-process.
static GLOBAL_CLIENT: LazyLock<Result<Client, String>> = LazyLock::new(|| {
    // Note that this is unsafe because it may misinterpret file descriptors
    // on Unix as jobserver file descriptors. We hopefully execute this near
    // the beginning of the process though to ensure we don't get false
    // positives, or in other words we try to execute this before we open
    // any file descriptors ourselves.
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

// Create a new jobserver if there's no inherited one.
fn default_client() -> Client {
    // Pick a "reasonable maximum" capping out at 32
    // so we don't take everything down by hogging the process run queue.
    // The fixed number is used to have deterministic compilation across machines.
    let client = Client::new(32).expect("failed to create jobserver");

    // Acquire a token for the main thread which we can release later
    client.acquire_raw().ok();

    client
}

static GLOBAL_CLIENT_CHECKED: OnceLock<Client> = OnceLock::new();

pub fn initialize_checked(report_warning: impl FnOnce(&'static str)) {
    let client_checked = match &*GLOBAL_CLIENT {
        Ok(client) => client.clone(),
        Err(e) => {
            report_warning(e);
            default_client()
        }
    };
    GLOBAL_CLIENT_CHECKED.set(client_checked).ok();
}

const ACCESS_ERROR: &str = "jobserver check should have been called earlier";

pub fn client() -> Client {
    GLOBAL_CLIENT_CHECKED.get().expect(ACCESS_ERROR).clone()
}

struct ProxyData {
    /// The number of tokens assigned to threads.
    /// If this is 0, a single token is still assigned to this process, but is unused.
    used: u16,

    /// The number of threads requesting a token
    pending: u16,
}

/// This is a jobserver proxy used to ensure that we hold on to at least one token.
pub struct Proxy {
    client: Client,
    data: Mutex<ProxyData>,

    /// Threads which are waiting on a token will wait on this.
    wake_pending: Condvar,

    helper: OnceLock<HelperThread>,
}

impl Proxy {
    pub fn new() -> Arc<Self> {
        let proxy = Arc::new(Proxy {
            client: client(),
            data: Mutex::new(ProxyData { used: 1, pending: 0 }),
            wake_pending: Condvar::new(),
            helper: OnceLock::new(),
        });
        let proxy_ = Arc::clone(&proxy);
        let helper = proxy
            .client
            .clone()
            .into_helper_thread(move |token| {
                if let Ok(token) = token {
                    let mut data = proxy_.data.lock();
                    if data.pending > 0 {
                        // Give the token to a waiting thread
                        token.drop_without_releasing();
                        assert!(data.used > 0);
                        data.used += 1;
                        data.pending -= 1;
                        proxy_.wake_pending.notify_one();
                    } else {
                        // The token is no longer needed, drop it.
                        drop(data);
                        drop(token);
                    }
                }
            })
            .expect("failed to create helper thread");
        proxy.helper.set(helper).unwrap();
        proxy
    }

    pub fn acquire_thread(&self) {
        let mut data = self.data.lock();

        if data.used == 0 {
            // There was a free token around. This can
            // happen when all threads release their token.
            assert_eq!(data.pending, 0);
            data.used += 1;
        } else {
            // Request a token from the helper thread. We can't directly use `acquire_raw`
            // as we also need to be able to wait for the final token in the process which
            // does not get a corresponding `release_raw` call.
            self.helper.get().unwrap().request_token();
            data.pending += 1;
            self.wake_pending.wait(&mut data);
        }
    }

    pub fn release_thread(&self) {
        let mut data = self.data.lock();

        if data.pending > 0 {
            // Give the token to a waiting thread
            data.pending -= 1;
            self.wake_pending.notify_one();
        } else {
            data.used -= 1;

            // Release the token unless it's the last one in the process
            if data.used > 0 {
                drop(data);
                self.client.release_raw().ok();
            }
        }
    }
}
