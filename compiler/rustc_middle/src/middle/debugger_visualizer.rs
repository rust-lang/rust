use std::path::PathBuf;
use std::sync::Arc;

use rustc_hir::attrs::DebuggerVisualizerType;
use rustc_macros::{Decodable, Encodable, HashStable};

/// A single debugger visualizer file.
#[derive(HashStable)]
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Encodable, Decodable)]
pub struct DebuggerVisualizerFile {
    /// The complete debugger visualizer source.
    pub src: Arc<[u8]>,
    /// Indicates which visualizer type this targets.
    pub visualizer_type: DebuggerVisualizerType,
    /// The file path to the visualizer file. This is used for reporting
    /// visualizer files in dep-info. Before it is written to crate metadata,
    /// the path is erased to `None`, so as not to emit potentially privacy
    /// sensitive data.
    pub path: Option<PathBuf>,
}

impl DebuggerVisualizerFile {
    pub fn new(src: Arc<[u8]>, visualizer_type: DebuggerVisualizerType, path: PathBuf) -> Self {
        DebuggerVisualizerFile { src, visualizer_type, path: Some(path) }
    }

    pub fn path_erased(&self) -> Self {
        DebuggerVisualizerFile {
            src: Arc::clone(&self.src),
            visualizer_type: self.visualizer_type,
            path: None,
        }
    }
}
