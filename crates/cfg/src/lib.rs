//! cfg defines conditional compiling options, `cfg` attibute parser and evaluator

mod cfg_expr;

use rustc_hash::FxHashSet;
use tt::SmolStr;

pub use cfg_expr::{CfgAtom, CfgExpr};

/// Configuration options used for conditional compilition on items with `cfg` attributes.
/// We have two kind of options in different namespaces: atomic options like `unix`, and
/// key-value options like `target_arch="x86"`.
///
/// Note that for key-value options, one key can have multiple values (but not none).
/// `feature` is an example. We have both `feature="foo"` and `feature="bar"` if features
/// `foo` and `bar` are both enabled. And here, we store key-value options as a set of tuple
/// of key and value in `key_values`.
///
/// See: https://doc.rust-lang.org/reference/conditional-compilation.html#set-configuration-options
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CfgOptions {
    enabled: FxHashSet<CfgAtom>,
}

impl CfgOptions {
    pub fn check(&self, cfg: &CfgExpr) -> Option<bool> {
        cfg.fold(&|atom| self.enabled.contains(atom))
    }

    pub fn insert_atom(&mut self, key: SmolStr) {
        self.enabled.insert(CfgAtom::Flag(key));
    }

    pub fn insert_key_value(&mut self, key: SmolStr, value: SmolStr) {
        self.enabled.insert(CfgAtom::KeyValue { key, value });
    }

    pub fn append(&mut self, other: &CfgOptions) {
        for atom in &other.enabled {
            self.enabled.insert(atom.clone());
        }
    }
}
