//! ra_cfg defines conditional compiling options, `cfg` attibute parser and evaluator
use ra_syntax::SmolStr;
use rustc_hash::{FxHashMap, FxHashSet};

mod cfg_expr;

pub use cfg_expr::{parse_cfg, CfgExpr};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CfgOptions {
    atoms: FxHashSet<SmolStr>,
    features: FxHashSet<SmolStr>,
    options: FxHashMap<SmolStr, SmolStr>,
}

impl CfgOptions {
    pub fn check(&self, cfg: &CfgExpr) -> Option<bool> {
        cfg.fold(&|key, value| match value {
            None => self.atoms.contains(key),
            Some(value) if key == "feature" => self.features.contains(value),
            Some(value) => self.options.get(key).map_or(false, |v| v == value),
        })
    }

    pub fn is_cfg_enabled(&self, attr: &tt::Subtree) -> Option<bool> {
        self.check(&parse_cfg(attr))
    }

    pub fn atom(mut self, name: SmolStr) -> CfgOptions {
        self.atoms.insert(name);
        self
    }

    pub fn feature(mut self, name: SmolStr) -> CfgOptions {
        self.features.insert(name);
        self
    }

    pub fn option(mut self, key: SmolStr, value: SmolStr) -> CfgOptions {
        self.options.insert(key, value);
        self
    }
}
