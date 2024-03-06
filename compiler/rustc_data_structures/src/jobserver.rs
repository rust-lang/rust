pub use jobserver_crate::Client;

use jobserver_crate::{FromEnv, FromEnvErrorKind};

use std::sync::{LazyLock, OnceLock};

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

pub fn acquire_thread() {
    GLOBAL_CLIENT_CHECKED.get().expect(ACCESS_ERROR).acquire_raw().ok();
}

pub fn release_thread() {
    GLOBAL_CLIENT_CHECKED.get().expect(ACCESS_ERROR).release_raw().ok();
}
