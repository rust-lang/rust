//! Parsing of CfgFlags as command line arguments, as in
//!
//! rustc main.rs --cfg foo --cfg 'feature="bar"'
use std::{fmt, str::FromStr};

use cfg::CfgOptions;

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum CfgFlag {
    Atom(String),
    KeyValue { key: String, value: String },
}

impl FromStr for CfgFlag {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let res = match s.split_once('=') {
            Some((key, value)) => {
                if !(value.starts_with('"') && value.ends_with('"')) {
                    return Err(format!("Invalid cfg ({s:?}), value should be in quotes"));
                }
                let key = key.to_string();
                let value = value[1..value.len() - 1].to_string();
                CfgFlag::KeyValue { key, value }
            }
            None => CfgFlag::Atom(s.into()),
        };
        Ok(res)
    }
}

impl<'de> serde::Deserialize<'de> for CfgFlag {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        String::deserialize(deserializer)?.parse().map_err(serde::de::Error::custom)
    }
}

impl Extend<CfgFlag> for CfgOptions {
    fn extend<T: IntoIterator<Item = CfgFlag>>(&mut self, iter: T) {
        for cfg_flag in iter {
            match cfg_flag {
                CfgFlag::Atom(it) => self.insert_atom(it.into()),
                CfgFlag::KeyValue { key, value } => self.insert_key_value(key.into(), value.into()),
            }
        }
    }
}

impl FromIterator<CfgFlag> for CfgOptions {
    fn from_iter<T: IntoIterator<Item = CfgFlag>>(iter: T) -> Self {
        let mut this = CfgOptions::default();
        this.extend(iter);
        this
    }
}

impl fmt::Display for CfgFlag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CfgFlag::Atom(atom) => f.write_str(atom),
            CfgFlag::KeyValue { key, value } => {
                f.write_str(key)?;
                f.write_str("=")?;
                f.write_str(value)
            }
        }
    }
}
