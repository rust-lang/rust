#![allow(fuzzy_provenance_casts)] // FIXME: this module systematically confuses pointers and integers

use crate::collections::HashMap;
use crate::ffi::{OsStr, OsString};
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sync::{Mutex, Once};
use crate::{fmt, io, vec};

// Specifying linkage/symbol name is solely to ensure a single instance between this crate and its unit tests
#[cfg_attr(test, linkage = "available_externally")]
#[unsafe(export_name = "_ZN16__rust_internals3std3sys3pal3sgx2os3ENVE")]
static ENV: AtomicUsize = AtomicUsize::new(0);
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

pub struct Env {
    iter: vec::IntoIter<(OsString, OsString)>,
}

// FIXME(https://github.com/rust-lang/rust/issues/114583): Remove this when <OsStr as Debug>::fmt matches <str as Debug>::fmt.
pub struct EnvStrDebug<'a> {
    slice: &'a [(OsString, OsString)],
}

impl fmt::Debug for EnvStrDebug<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { slice } = self;
        f.debug_list()
            .entries(slice.iter().map(|(a, b)| (a.to_str().unwrap(), b.to_str().unwrap())))
            .finish()
    }
}

impl Env {
    pub fn str_debug(&self) -> impl fmt::Debug + '_ {
        let Self { iter } = self;
        EnvStrDebug { slice: iter.as_slice() }
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { iter } = self;
        f.debug_list().entries(iter.as_slice()).finish()
    }
}

impl !Send for Env {}
impl !Sync for Env {}

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub fn env() -> Env {
    let clone_to_vec = |map: &HashMap<OsString, OsString>| -> Vec<_> {
        map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    };

    let iter = get_env_store()
        .map(|env| clone_to_vec(&env.lock().unwrap()))
        .unwrap_or_default()
        .into_iter();
    Env { iter }
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
