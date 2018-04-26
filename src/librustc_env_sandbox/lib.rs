// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![deny(warnings)]

#[cfg(test)]
extern crate tempdir;

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

// Set up a sandbox to access the compiler's system environment. This includes:
// - what appears in the environment (ie, visible to the `env!()` and `option_env!()` macros)
// - constrain the path prefixs files may be included from

/// Aspects of the compiler's environment which can be sandboxed. If the field is `None`,
/// is unconstrained.
#[derive(Debug, Clone, Default)]
pub struct EnvSandbox {
    /// Set of environment variables available. If a variable appears with a `None` value
    /// then it means the variable can be read from the real system environment. Otherwise
    /// the variable is as it has been defined here.
    env: Option<HashMap<String, Option<String>>>,
    /// Set of paths from which files can be included. These are normalized to the real path,
    /// so all files being opened are also opened via normalized real paths.
    paths: Option<Vec<PathBuf>>,
}

/// Builder for an `EnvSandbox`
pub struct EnvSandboxBuilder(EnvSandbox);

impl EnvSandboxBuilder {
    /// Construct a new `EnvSandboxBuilder`.
    pub fn new() -> Self {
        EnvSandboxBuilder(EnvSandbox {
            env: None,
            paths: None,
        })
    }

    fn env_add_filter<K, V>(&mut self, var: K, val: Option<V>)
    where
        String: From<K>,
        String: From<V>,
    {
        let var = String::from(var);
        let val = val.map(String::from);

        let mut env = self.0.env.take().unwrap_or(HashMap::new());

        env.insert(var, val);

        self.0.env = Some(env);
    }

    /// Define a new name->value mapping
    pub fn env_define<K, V>(&mut self, var: K, val: V) -> &mut Self
    where
        String: From<K>,
        String: From<V>,
    {
        self.env_add_filter(var, Some(val));
        self
    }

    /// Allow a give name to be accessed from the environment
    pub fn env_allow<K>(&mut self, var: K) -> &mut Self
    where
        String: From<K>,
    {
        self.env_add_filter::<_, String>(var, None);
        self
    }

    /// Set up an empty mapping, prohibiting access to the process environment without
    /// defining any new variables.
    pub fn env_clear(&mut self) -> &mut Self {
        self.0.env = Some(HashMap::new());
        self
    }

    /// Clear all path prefixes, leaving no valid prefixes. This prevents all file access.
    pub fn paths_clear(&mut self) -> &mut Self {
        self.0.paths = Some(Vec::new());
        self
    }

    /// Add a path prefix to the allowed set. The path must exist so that it can
    /// be canonicalized.
    pub fn path_add<P: AsRef<Path>>(&mut self, path: P) -> io::Result<&mut Self> {
        let path = path.as_ref();
        let path = fs::canonicalize(path)?;

        let mut paths = self.0.paths.take().unwrap_or(Vec::new());
        paths.push(path);
        self.0.paths = Some(paths);

        Ok(self)
    }

    /// Construct an `EnvSandbox` from the builder
    pub fn build(self) -> EnvSandbox {
        self.0
    }
}

impl EnvSandbox {
    /// Get an environment variable, either from the real environment
    /// or from the locally defined sandbox variables.
    pub fn env_get(&self, var: &str) -> Option<String> {
        match &self.env {
            &None => env::var(var).ok(),
            &Some(ref map) => match map.get(var) {
                None => None,
                Some(&Some(ref val)) => Some(val.to_string()),
                Some(&None) => env::var(var).ok(),
            },
        }
    }

    /// Return true if a given path has a valid prefix. Fails if the path
    /// can't be canonicalized.
    pub fn path_ok<P: AsRef<Path>>(&self, path: P) -> io::Result<bool> {
        let path = path.as_ref();
        let fullpath = fs::canonicalize(path)?;

        let ret = if let Some(ref paths) = self.paths {
            paths.iter().any(|p| fullpath.starts_with(p))
        } else {
            true
        };
        Ok(ret)
    }

    /// Map a path to itself if it has a valid prefix, otherwise return a suitable
    /// error.
    pub fn path_lookup<P>(&self, path: P) -> io::Result<P>
    where
        P: AsRef<Path>,
    {
        if self.path_ok(&path)? {
            Ok(path)
        } else {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "path does not have a valid prefix",
            ))
        }
    }

    /// Open a path for reading if it has a valid prefix.
    pub fn path_open<P: AsRef<Path>>(&self, path: P) -> io::Result<File> {
        File::open(self.path_lookup(path)?)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use std::env;
    use tempdir::TempDir;

    #[test]
    fn test_env() {
        env::set_var("ZUBZUB", "ZIBZIB");

        let sb = EnvSandbox::default();

        assert_eq!(env::var("ZUBZUB"), Ok("ZIBZIB".to_string()));
        assert_eq!(sb.env_get("ZUBZUB"), Some("ZIBZIB".to_string()));
    }

    #[test]
    fn test_env_filter() {
        let mut sb = EnvSandboxBuilder::new();

        sb.env_define("FOO", "BAR");

        let sb = sb.build();

        assert_eq!(sb.env_get("FOO"), Some("BAR".to_string()));
        assert_eq!(sb.env_get("BLAHBLAH"), None);
    }

    #[test]
    fn test_env_allow() {
        let mut sb = EnvSandboxBuilder::new();

        sb.env_define("FOO", "BAR")
            .env_allow("BLOP")
            .env_allow("GLUB");

        let sb = sb.build();

        env::set_var("ZUBZUB", "ZIBZIB");
        env::set_var("BLOP", "Blub");
        env::remove_var("GLUB");

        assert_eq!(sb.env_get("FOO"), Some("BAR".to_string()));
        assert_eq!(sb.env_get("ZUBZUB"), None);
        assert_eq!(sb.env_get("BLOP"), Some("Blub".to_string()));
        assert_eq!(sb.env_get("GLUB"), None);
    }

    #[test]
    fn test_env_clear() {
        let mut sb = EnvSandboxBuilder::new();

        sb.env_define("FOO", "BAR")
            .env_allow("BLOP")
            .env_allow("GLUB")
            .env_clear();

        let sb = sb.build();

        env::set_var("ZUBZUB", "ZIBZIB");
        env::set_var("BLOP", "Blub");
        env::remove_var("GLUB");

        assert_eq!(sb.env_get("FOO"), None);
        assert_eq!(sb.env_get("ZUBZUB"), None);
        assert_eq!(sb.env_get("BLOP"), None);
        assert_eq!(sb.env_get("GLUB"), None);
    }

    #[test]
    fn test_path() {
        let dir = TempDir::new("test").expect("tempdir failed");
        let subdir = dir.path().join("subdir");
        fs::create_dir(&subdir).expect("subdir failed");

        let sb = EnvSandbox::default();

        assert!(sb.path_ok(&dir).unwrap());
        assert!(sb.path_ok(dir.path().parent().unwrap()).unwrap());
        assert!(sb.path_ok(&subdir).unwrap());
        assert!(sb.path_ok("/").unwrap());
    }

    #[test]
    fn test_path_filter() {
        let dir = TempDir::new("test").expect("tempdir failed");
        let subdir = dir.path().join("subdir");
        fs::create_dir(&subdir).expect("subdir failed");

        let mut sb = EnvSandboxBuilder::new();
        sb.path_add(&dir).unwrap();
        let sb = sb.build();

        assert!(sb.path_ok(&dir).unwrap());
        assert!(!sb.path_ok(dir.path().parent().unwrap()).unwrap());
        assert!(sb.path_ok(&subdir).unwrap());
        assert!(!sb.path_ok("/").unwrap());
    }

    #[test]
    fn test_path_clear() {
        let dir = TempDir::new("test").expect("tempdir failed");
        let subdir = dir.path().join("subdir");
        fs::create_dir(&subdir).expect("subdir failed");

        let mut sb = EnvSandboxBuilder::new();
        sb.path_add(&dir).unwrap();
        sb.paths_clear();
        let sb = sb.build();

        assert!(!sb.path_ok(&dir).unwrap());
        assert!(!sb.path_ok(dir.path().parent().unwrap()).unwrap());
        assert!(!sb.path_ok(&subdir).unwrap());
        assert!(!sb.path_ok("/").unwrap());
    }
}
