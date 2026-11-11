//! Stuff for handling `#[target_feature]` (needed for unsafe check).

use std::sync::LazyLock;
use std::{borrow::Cow, ops::Deref};

use hir_def::FunctionId;
use hir_def::attrs::AttrFlags;
use intern::Symbol;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::db::HirDatabase;
use generated::TARGET_FEATURE_IMPLICATIONS_RAW;

mod generated;

#[derive(Debug, Default, Clone)]
pub struct TargetFeatures<'db> {
    pub(crate) enabled: Cow<'db, FxHashSet<Symbol>>,
}

impl<'db> TargetFeatures<'db> {
    pub fn from_fn(db: &'db dyn HirDatabase, owner: FunctionId) -> Self {
        let mut result = TargetFeatures::from_fn_no_implications(db, owner);
        result.expand_implications();
        result
    }

    fn expand_implications(&mut self) {
        let all_implications = LazyLock::force(&TARGET_FEATURE_IMPLICATIONS);
        let enabled = self.enabled.to_mut();
        let mut queue = enabled.iter().cloned().collect::<Vec<_>>();
        while let Some(feature) = queue.pop() {
            if let Some(implications) = all_implications.get(&feature) {
                for implication in implications {
                    if enabled.insert(implication.clone()) {
                        queue.push(implication.clone());
                    }
                }
            }
        }
    }

    /// Retrieves the target features from the attributes, and does not expand the target features implied by them.
    pub(crate) fn from_fn_no_implications(db: &'db dyn HirDatabase, owner: FunctionId) -> Self {
        let enabled = AttrFlags::target_features(db, owner);
        Self { enabled: Cow::Borrowed(enabled) }
    }
}

// List of the target features each target feature implies.
// Ideally we'd depend on rustc for this, but rustc_target doesn't compile on stable,
// and t-compiler prefers for it to stay this way.

static TARGET_FEATURE_IMPLICATIONS: LazyLock<FxHashMap<Symbol, Box<[Symbol]>>> =
    LazyLock::new(|| {
        let mut result = FxHashMap::<Symbol, FxHashSet<Symbol>>::default();
        for &(feature_str, ref _stability, implications) in
            TARGET_FEATURE_IMPLICATIONS_RAW.iter().flat_map(Deref::deref)
        {
            let feature = Symbol::intern(feature_str);
            let implications = implications.iter().copied().map(Symbol::intern);
            // Some target features appear in two archs, e.g. Arm and x86.
            // Sometimes they contain different implications, e.g. `aes`.
            // We should probably choose by the active arch, but for now just merge them.
            result.entry(feature).or_default().extend(implications);
        }
        let mut result = result
            .into_iter()
            .map(|(feature, implications)| (feature, Box::from_iter(implications)))
            .collect::<FxHashMap<_, _>>();
        result.shrink_to_fit();
        result
    });
