use jobserver::Client;
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

// Unlike releasing tokens, there's not really a "one size fits all" approach, as we have two
// primary ways of acquiring a token: via the helper thread, and via the acquire_thread function.
//
// That makes this function necessary unlike in the release case where everything is piped through
// `release_thread`.
fn notify_acquiring_token() {
    // this function does nothing for now but will be wired up to send a message to Cargo
}

pub fn initialize() {
    lazy_static::initialize(&GLOBAL_CLIENT)
}

pub struct HelperThread {
    helper: jobserver_crate::HelperThread,
}

impl HelperThread {
    // This API does not block, but is shimmed so that we can inform Cargo of our interest here.
    pub fn request_token(&self) {
        notify_acquiring_token();
        self.helper.request_token();
    }
}

pub struct Acquired(());

impl Drop for Acquired {
    fn drop(&mut self) {
        release_thread();
    }
}

pub fn helper_thread<F>(mut cb: F) -> HelperThread
where
    F: FnMut(Acquired) + Send + 'static,
{
    let thread = GLOBAL_CLIENT
        .clone()
        .into_helper_thread(move |token| {
            // we've acquired a token, but we need to not use it as we have our own
            // custom release-on-drop struct since we'll want different logic than
            // just normally releasing the token in this case.
            //
            // On unix this unfortunately means that we lose the specific byte that
            // was in the pipe (i.e., we just write back the same byte all the time)
            // but that's not expected to be a problem.
            std::mem::forget(token.expect("acquire token"));
            cb(Acquired(()))
        })
        .expect("failed to spawn helper thread");

    HelperThread { helper: thread }
}

pub fn acquire_thread() {
    notify_acquiring_token();
    GLOBAL_CLIENT.acquire_raw().ok();
}

pub fn release_thread() {
    GLOBAL_CLIENT.release_raw().ok();
}
