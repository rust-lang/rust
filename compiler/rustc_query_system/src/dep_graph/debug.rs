//! Code for debugging the dep-graph.

use super::{DepNode, DepNodeIndex};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::Lock;
use std::error::Error;

/// A dep-node filter goes from a user-defined string to a query over
/// nodes. Right now the format is like this:
/// ```ignore (illustrative)
/// x & y & z
/// ```
/// where the format-string of the dep-node must contain `x`, `y`, and
/// `z`.
#[derive(Debug)]
pub struct DepNodeFilter {
    text: String,
}

impl DepNodeFilter {
    pub fn new(text: &str) -> Self {
        DepNodeFilter { text: text.trim().to_string() }
    }

    /// Returns `true` if all nodes always pass the filter.
    pub fn accepts_all(&self) -> bool {
        self.text.is_empty()
    }

    /// Tests whether `node` meets the filter, returning true if so.
    pub fn test(&self, node: &DepNode) -> bool {
        let debug_str = format!("{node:?}");
        self.text.split('&').map(|s| s.trim()).all(|f| debug_str.contains(f))
    }
}

/// A filter like `F -> G` where `F` and `G` are valid dep-node
/// filters. This can be used to test the source/target independently.
pub struct EdgeFilter {
    pub source: DepNodeFilter,
    pub target: DepNodeFilter,
    pub index_to_node: Lock<FxHashMap<DepNodeIndex, DepNode>>,
}

impl EdgeFilter {
    pub fn new(test: &str) -> Result<EdgeFilter, Box<dyn Error>> {
        let parts: Vec<_> = test.split("->").collect();
        if parts.len() != 2 {
            Err(format!("expected a filter like `a&b -> c&d`, not `{test}`").into())
        } else {
            Ok(EdgeFilter {
                source: DepNodeFilter::new(parts[0]),
                target: DepNodeFilter::new(parts[1]),
                index_to_node: Lock::new(FxHashMap::default()),
            })
        }
    }

    #[cfg(debug_assertions)]
    pub fn test(&self, source: &DepNode, target: &DepNode) -> bool {
        self.source.test(source) && self.target.test(target)
    }
}
