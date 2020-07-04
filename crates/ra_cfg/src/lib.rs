//! ra_cfg defines conditional compiling options, `cfg` attibute parser and evaluator

mod cfg_expr;

use ra_syntax::SmolStr;
use rustc_hash::FxHashSet;

pub use cfg_expr::{parse_cfg, CfgExpr};

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
    atoms: FxHashSet<SmolStr>,
    key_values: FxHashSet<(SmolStr, SmolStr)>,
}

impl CfgOptions {
    pub fn check(&self, cfg: &CfgExpr) -> Option<bool> {
        cfg.fold(&|key, value| match value {
            None => self.atoms.contains(key),
            Some(value) => self.key_values.contains(&(key.clone(), value.clone())),
        })
    }

    pub fn is_cfg_enabled(&self, attr: &tt::Subtree) -> Option<bool> {
        self.check(&parse_cfg(attr))
    }

    pub fn insert_atom(&mut self, key: SmolStr) {
        self.atoms.insert(key);
    }

    pub fn remove_atom(&mut self, name: &str) {
        self.atoms.remove(name);
    }

    pub fn insert_key_value(&mut self, key: SmolStr, value: SmolStr) {
        self.key_values.insert((key, value));
    }

    pub fn append(&mut self, other: &CfgOptions) {
        for atom in &other.atoms {
            self.atoms.insert(atom.clone());
        }

        for (key, value) in &other.key_values {
            self.key_values.insert((key.clone(), value.clone()));
        }
    }
}
