#![allow(dead_code)]
#![unstable(feature = "process_internals", issue = "0")]

use crate::collections::BTreeMap;
use crate::env;
use crate::ffi::{OsStr, OsString};
use crate::sys::process::EnvKey;

// Stores a set of changes to an environment
#[derive(Clone, Debug)]
pub struct CommandEnv {
    clear: bool,
    saw_path: bool,
    vars: BTreeMap<EnvKey, Option<OsString>>,
}

impl Default for CommandEnv {
    fn default() -> Self {
        CommandEnv { clear: false, saw_path: false, vars: Default::default() }
    }
}

impl CommandEnv {
    // Capture the current environment with these changes applied
    pub fn capture(&self) -> BTreeMap<EnvKey, OsString> {
        let mut result = BTreeMap::<EnvKey, OsString>::new();
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

    pub fn capture_if_changed(&self) -> Option<BTreeMap<EnvKey, OsString>> {
        if self.is_unchanged() { None } else { Some(self.capture()) }
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
