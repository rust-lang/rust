pub use super::common::Env;
use crate::collections::HashMap;
use crate::ffi::{OsStr, OsString};
use crate::io;
use crate::sync::{Mutex, OnceLock};
use crate::sys::pal::params;

type EnvStore = Mutex<HashMap<OsString, OsString>>;

static ENV: OnceLock<EnvStore> = OnceLock::new();

fn get_env_store() -> &'static EnvStore {
    ENV.get_or_init(|| {
        let mut env_store = HashMap::new();
        if let Some(params) = params::get() {
            for param in params {
                if let Ok(envs) = params::EnvironmentBlock::try_from(&param) {
                    for env in envs {
                        env_store.insert(env.key.into(), env.value.into());
                    }
                    break;
                }
            }
        }
        Mutex::new(env_store)
    })
}

pub fn env() -> Env {
    let env = get_env_store().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
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
