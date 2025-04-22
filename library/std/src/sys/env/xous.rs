use crate::collections::HashMap;
use crate::ffi::{OsStr, OsString};
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sync::{Mutex, Once};
use crate::sys::pal::os::{get_application_parameters, params};
use crate::{fmt, io, vec};

static ENV: AtomicUsize = AtomicUsize::new(0);
static ENV_INIT: Once = Once::new();
type EnvStore = Mutex<HashMap<OsString, OsString>>;

fn get_env_store() -> &'static EnvStore {
    ENV_INIT.call_once(|| {
        let env_store = EnvStore::default();
        if let Some(params) = get_application_parameters() {
            for param in params {
                if let Ok(envs) = params::EnvironmentBlock::try_from(&param) {
                    let mut env_store = env_store.lock().unwrap();
                    for env in envs {
                        env_store.insert(env.key.into(), env.value.into());
                    }
                    break;
                }
            }
        }
        ENV.store(Box::into_raw(Box::new(env_store)) as _, Ordering::Relaxed)
    });
    unsafe { &*core::ptr::with_exposed_provenance::<EnvStore>(ENV.load(Ordering::Relaxed)) }
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
    // FIXME(https://github.com/rust-lang/rust/issues/114583): Remove this when <OsStr as Debug>::fmt matches <str as Debug>::fmt.
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

    let iter = clone_to_vec(&*get_env_store().lock().unwrap()).into_iter();
    Env { iter }
}

pub fn getenv(k: &OsStr) -> Option<OsString> {
    get_env_store().lock().unwrap().get(k).cloned()
}

pub unsafe fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    let (k, v) = (k.to_owned(), v.to_owned());
    get_env_store().lock().unwrap().insert(k, v);
    Ok(())
}

pub unsafe fn unsetenv(k: &OsStr) -> io::Result<()> {
    get_env_store().lock().unwrap().remove(k);
    Ok(())
}
