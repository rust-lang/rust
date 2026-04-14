//! Artifact collection for BDD test runs.
//!
//! Manages screenshots, serial logs, and generates markdown reports.
//! Outputs to `/docs/behavior/${ARCH}/${FEATURE}/${SCENARIO}/${STEP}/`
//! with markdown summaries at each level.

use std::sync::OnceLock;
use tokio::sync::Mutex;

mod collector;
pub mod qmp;
mod types;

pub use collector::*;
pub use qmp::*;
pub use types::*;

/// Global artifact collector instance.
static COLLECTOR: OnceLock<Mutex<ArtifactCollector>> = OnceLock::new();

/// Global serial log cache (updated by world, read by reporter).
static SERIAL_LOG: OnceLock<Mutex<String>> = OnceLock::new();

/// Initialize the global artifact collector for the given architecture.
pub fn init_global(arch: &str) {
    let collector = ArtifactCollector::new(arch);
    let _ = collector.init();
    let _ = COLLECTOR.set(Mutex::new(collector));
    let _ = SERIAL_LOG.set(Mutex::new(String::new()));
    let _ = QMP_STREAM.set(Mutex::new(None));
}

/// Get the global artifact collector.
pub fn global() -> &'static Mutex<ArtifactCollector> {
    COLLECTOR
        .get()
        .expect("ArtifactCollector not initialized - call init_global first")
}

/// Update the global serial log cache (called from world).
pub async fn set_latest_serial(log: &str) {
    if let Some(cache) = SERIAL_LOG.get() {
        let mut serial = cache.lock().await;
        *serial = log.to_string();
    }
}

/// Get the latest serial log (for reporter to use).
pub async fn get_latest_serial() -> String {
    if let Some(cache) = SERIAL_LOG.get() {
        cache.lock().await.clone()
    } else {
        String::new()
    }
}
