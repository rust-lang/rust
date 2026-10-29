pub use super::common::Env;
use crate::collections::HashMap;
use crate::ffi::{OsStr, OsString};
use crate::io;
use crate::sync::{Mutex, OnceLock};

type EnvStore = Mutex<HashMap<OsString, OsString>>;

// Specifying linkage/symbol name is solely to ensure a single instance between this crate and its unit tests
#[cfg_attr(test, linkage = "available_externally")]
#[unsafe(export_name = "_ZN16__rust_internals3std3sys3pal3sgx2os3ENVE")]
static ENV: OnceLock<EnvStore> = OnceLock::new();

pub fn env() -> Env {
    let env = ENV
        .get()
        .map(|env| env.lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    Env::new(env)
}

pub fn getenv(k: &OsStr) -> Option<OsString> {
    ENV.get().and_then(|s| s.lock().unwrap().get(k).cloned())
}

pub unsafe fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    ENV.get_or_init(|| EnvStore::default()).lock().unwrap().insert(k.to_owned(), v.to_owned());
    Ok(())
}

pub unsafe fn unsetenv(k: &OsStr) -> io::Result<()> {
    if let Some(env) = ENV.get() {
        env.lock().unwrap().remove(k);
    }
    Ok(())
}
