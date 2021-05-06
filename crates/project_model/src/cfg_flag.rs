//! Parsing of CfgFlags as command line arguments, as in
//!
//! rustc main.rs --cfg foo --cfg 'feature="bar"'
use std::str::FromStr;

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
                    return Err(format!("Invalid cfg ({:?}), value should be in quotes", s));
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
