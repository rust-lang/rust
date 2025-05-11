#![allow(fuzzy_provenance_casts)] // FIXME: this module systematically confuses pointers and integers

pub use super::common::Env;
use crate::collections::HashMap;
use crate::ffi::{OsStr, OsString};
use crate::io;
use crate::sync::atomic::{Atomic, AtomicUsize, Ordering};
use crate::sync::{Mutex, Once};

// Specifying linkage/symbol name is solely to ensure a single instance between this crate and its unit tests
#[cfg_attr(test, linkage = "available_externally")]
#[unsafe(export_name = "_ZN16__rust_internals3std3sys3pal3sgx2os3ENVE")]
static ENV: Atomic<usize> = AtomicUsize::new(0);
// Specifying linkage/symbol name is solely to ensure a single instance between this crate and its unit tests
#[cfg_attr(test, linkage = "available_externally")]
#[unsafe(export_name = "_ZN16__rust_internals3std3sys3pal3sgx2os8ENV_INITE")]
static ENV_INIT: Once = Once::new();
type EnvStore = Mutex<HashMap<OsString, OsString>>;

fn get_env_store() -> Option<&'static EnvStore> {
    unsafe { (ENV.load(Ordering::Relaxed) as *const EnvStore).as_ref() }
}

fn create_env_store() -> &'static EnvStore {
    ENV_INIT.call_once(|| {
        ENV.store(Box::into_raw(Box::new(EnvStore::default())) as _, Ordering::Relaxed)
    });
    unsafe { &*(ENV.load(Ordering::Relaxed) as *const EnvStore) }
}

pub fn env() -> Env {
    let clone_to_vec = |map: &HashMap<OsString, OsString>| -> Vec<_> {
        map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    };

    let env = get_env_store().map(|env| clone_to_vec(&env.lock().unwrap())).unwrap_or_default();
    Env::new(env)
}

pub fn getenv(k: &OsStr) -> Option<OsString> {
    get_env_store().and_then(|s| s.lock().unwrap().get(k).cloned())
}

pub unsafe fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    let (k, v) = (k.to_owned(), v.to_owned());
    create_env_store().lock().unwrap().insert(k, v);
    Ok(())
}

pub unsafe fn unsetenv(k: &OsStr) -> io::Result<()> {
    if let Some(env) = get_env_store() {
        env.lock().unwrap().remove(k);
    }
    Ok(())
}
