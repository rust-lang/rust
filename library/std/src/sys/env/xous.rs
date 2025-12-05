pub use super::common::Env;
use crate::collections::HashMap;
use crate::ffi::{OsStr, OsString};
use crate::io;
use crate::sync::atomic::{Atomic, AtomicUsize, Ordering};
use crate::sync::{Mutex, Once};
use crate::sys::pal::os::{get_application_parameters, params};

static ENV: Atomic<usize> = AtomicUsize::new(0);
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

pub fn env() -> Env {
    let clone_to_vec = |map: &HashMap<OsString, OsString>| -> Vec<_> {
        map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    };

    let env = clone_to_vec(&*get_env_store().lock().unwrap());
    Env::new(env)
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
