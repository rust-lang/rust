use jobserver_crate::{Client, HelperThread, Acquired};
use lazy_static::lazy_static;
use std::sync::{Condvar, Arc, Mutex};
use std::mem;

#[derive(Default)]
struct LockedProxyData {
    /// The number of free thread tokens, this may include the implicit token given to the process
    free: usize,

    /// The number of threads waiting for a token
    waiters: usize,

    /// The number of tokens we requested from the server
    requested: usize,

    /// Stored tokens which will be dropped when we no longer need them
    tokens: Vec<Acquired>,
}

impl LockedProxyData {
    fn request_token(&mut self, thread: &Mutex<HelperThread>) {
        self.requested += 1;
        thread.lock().unwrap().request_token();
    }

    fn release_token(&mut self, cond_var: &Condvar) {
        if self.waiters > 0 {
            self.free += 1;
            cond_var.notify_one();
        } else {
            if self.tokens.is_empty() {
                // We are returning the implicit token
                self.free += 1;
            } else {
                // Return a real token to the server
                self.tokens.pop().unwrap();
            }
        }
    }

    fn take_token(&mut self, thread: &Mutex<HelperThread>) -> bool {
        if self.free > 0 {
            self.free -= 1;
            self.waiters -= 1;

            // We stole some token reqested by someone else
            // Request another one
            if self.requested + self.free < self.waiters {
                self.request_token(thread);
            }

            true
        } else {
            false
        }
    }

    fn new_requested_token(&mut self, token: Acquired, cond_var: &Condvar) {
        self.requested -= 1;

        // Does anything need this token?
        if self.waiters > 0 {
            self.free += 1;
            self.tokens.push(token);
            cond_var.notify_one();
        } else {
            // Otherwise we'll just drop it
            mem::drop(token);
        }
    }
}

#[derive(Default)]
struct ProxyData {
    lock: Mutex<LockedProxyData>,
    cond_var: Condvar,
}

/// A helper type which makes managing jobserver tokens easier.
/// It also allows you to treat the implicit token given to the process
/// in the same manner as requested tokens.
struct Proxy {
    thread: Mutex<HelperThread>,
    data: Arc<ProxyData>,
}

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
            Client::new(32).expect("failed to create jobserver")
        })
    };

    static ref GLOBAL_PROXY: Proxy = {
        let data = Arc::new(ProxyData::default());

        Proxy {
            data: data.clone(),
            thread: Mutex::new(client().into_helper_thread(move |token| {
                data.lock.lock().unwrap().new_requested_token(token.unwrap(), &data.cond_var);
            }).unwrap()),
        }
    };
}

pub fn client() -> Client {
    GLOBAL_CLIENT.clone()
}

pub fn acquire_thread() {
    GLOBAL_PROXY.acquire_token();
}

pub fn release_thread() {
    GLOBAL_PROXY.release_token();
}

impl Proxy {
    fn release_token(&self) {
        self.data.lock.lock().unwrap().release_token(&self.data.cond_var);
    }

    fn acquire_token(&self) {
        let mut data = self.data.lock.lock().unwrap();
        data.waiters += 1;
        if data.take_token(&self.thread) {
            return;
        }
        // Request a token for us
        data.request_token(&self.thread);
        loop {
            data = self.data.cond_var.wait(data).unwrap();
            if data.take_token(&self.thread) {
                return;
            }
        }
    }
}
