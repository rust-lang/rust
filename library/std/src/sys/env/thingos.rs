//! ThingOS environment variable implementation.
//!
//! Environment variables are stored in an in-process hash map.  On program
//! startup the kernel provides an environment block (as part of the process
//! ABI); subsequent mutations are local to the process.
//!
//! TODO: parse initial env from the kernel-supplied env block once the ABI
//! is finalised.

pub use super::common::Env;
use crate::collections::HashMap;
use crate::ffi::{OsStr, OsString};
use crate::io;
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sync::{Mutex, Once};

static ENV_STORE_PTR: AtomicUsize = AtomicUsize::new(0);
static ENV_INIT: Once = Once::new();
type EnvStore = Mutex<HashMap<OsString, OsString>>;

fn get_env_store() -> &'static EnvStore {
    ENV_INIT.call_once(|| {
        let store = EnvStore::default();
        // TODO: populate from kernel-supplied environment block.
        ENV_STORE_PTR.store(Box::into_raw(Box::new(store)) as _, Ordering::Relaxed);
    });
    // SAFETY: The pointer was stored exactly once by `call_once` above.
    unsafe { &*core::ptr::with_exposed_provenance::<EnvStore>(ENV_STORE_PTR.load(Ordering::Relaxed)) }
}

pub fn env() -> Env {
    let snapshot: Vec<(OsString, OsString)> = get_env_store()
        .lock()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Env::new(snapshot)
}

pub fn getenv(key: &OsStr) -> Option<OsString> {
    get_env_store().lock().unwrap().get(key).cloned()
}

pub unsafe fn setenv(key: &OsStr, val: &OsStr) -> io::Result<()> {
    let (k, v) = (key.to_owned(), val.to_owned());
    get_env_store().lock().unwrap().insert(k, v);
    Ok(())
}

pub unsafe fn unsetenv(key: &OsStr) -> io::Result<()> {
    get_env_store().lock().unwrap().remove(key);
    Ok(())
}
