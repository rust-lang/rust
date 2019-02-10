#![allow(dead_code)]
#![unstable(feature = "process_internals", issue = "0")]

use crate::ffi::{OsStr, OsString};
use crate::env;
use crate::collections::BTreeMap;
use crate::borrow::Borrow;

pub trait EnvKey:
    From<OsString> + Into<OsString> +
    Borrow<OsStr> + Borrow<Self> + AsRef<OsStr> +
    Ord + Clone {}

// Implement a case-sensitive environment variable key
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct DefaultEnvKey(OsString);

impl From<OsString> for DefaultEnvKey {
    fn from(k: OsString) -> Self { DefaultEnvKey(k) }
}

impl From<DefaultEnvKey> for OsString {
    fn from(k: DefaultEnvKey) -> Self { k.0 }
}

impl Borrow<OsStr> for DefaultEnvKey {
    fn borrow(&self) -> &OsStr { &self.0 }
}

impl AsRef<OsStr> for DefaultEnvKey {
    fn as_ref(&self) -> &OsStr { &self.0 }
}

impl EnvKey for DefaultEnvKey {}

// Stores a set of changes to an environment
#[derive(Clone, Debug)]
pub struct CommandEnv<K> {
    clear: bool,
    saw_path: bool,
    vars: BTreeMap<K, Option<OsString>>
}

impl<K: EnvKey> Default for CommandEnv<K> {
    fn default() -> Self {
        CommandEnv {
            clear: false,
            saw_path: false,
            vars: Default::default()
        }
    }
}

impl<K: EnvKey> CommandEnv<K> {
    // Capture the current environment with these changes applied
    pub fn capture(&self) -> BTreeMap<K, OsString> {
        let mut result = BTreeMap::<K, OsString>::new();
        if !self.clear {
            for (k, v) in env::vars_os() {
                result.insert(k.into(), v);
            }
        }
        for (k, maybe_v) in &self.vars {
            if let &Some(ref v) = maybe_v {
                result.insert(k.clone(), v.clone());
            } else {
                result.remove(k);
            }
        }
        result
    }

    // Apply these changes directly to the current environment
    pub fn apply(&self) {
        if self.clear {
            for (k, _) in env::vars_os() {
                env::remove_var(k);
            }
        }
        for (key, maybe_val) in self.vars.iter() {
            if let &Some(ref val) = maybe_val {
                env::set_var(key, val);
            } else {
                env::remove_var(key);
            }
        }
    }

    pub fn is_unchanged(&self) -> bool {
        !self.clear && self.vars.is_empty()
    }

    pub fn capture_if_changed(&self) -> Option<BTreeMap<K, OsString>> {
        if self.is_unchanged() {
            None
        } else {
            Some(self.capture())
        }
    }

    // The following functions build up changes
    pub fn set(&mut self, key: &OsStr, value: &OsStr) {
        self.maybe_saw_path(&key);
        self.vars.insert(key.to_owned().into(), Some(value.to_owned()));
    }
    pub fn remove(&mut self, key: &OsStr) {
        self.maybe_saw_path(&key);
        if self.clear {
            self.vars.remove(key);
        } else {
            self.vars.insert(key.to_owned().into(), None);
        }
    }
    pub fn clear(&mut self) {
        self.clear = true;
        self.vars.clear();
    }
    pub fn have_changed_path(&self) -> bool {
        self.saw_path || self.clear
    }
    fn maybe_saw_path(&mut self, key: &OsStr) {
        if !self.saw_path && key == "PATH" {
            self.saw_path = true;
        }
    }
}
