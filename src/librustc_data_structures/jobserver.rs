use jobserver_crate::Client;
use lazy_static::lazy_static;

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
    // The fixed number is used to have deterministic compilation
    // across machines.
    //
    // Also note that we stick this in a global because there could be
    // multiple rustc instances in this process, and the jobserver is
    // per-process.
    static ref GLOBAL_CLIENT: Client = unsafe {
        Client::from_env().unwrap_or_else(|| {
            let client = Client::new(32).expect("failed to create jobserver");
            // Acquire a token for the main thread which we can release later
            client.acquire_raw().ok();
            client
        })
    };
}

pub fn client() -> Client {
    GLOBAL_CLIENT.clone()
}

pub fn acquire_thread() {
    GLOBAL_CLIENT.acquire_raw().ok();
}

pub fn release_thread() {
    GLOBAL_CLIENT.release_raw().ok();
}
