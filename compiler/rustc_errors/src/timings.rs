use std::time::Instant;

use crate::DiagCtxtHandle;

/// A high-level section of the compilation process.
#[derive(Copy, Clone, Debug)]
pub enum TimingSection {
    /// Time spent linking.
    Linking,
}

/// Section with attached timestamp
#[derive(Copy, Clone, Debug)]
pub struct TimingRecord {
    pub section: TimingSection,
    /// Microseconds elapsed since some predetermined point in time (~start of the rustc process).
    pub timestamp: u128,
}

impl TimingRecord {
    fn from_origin(origin: Instant, section: TimingSection) -> Self {
        Self { section, timestamp: Instant::now().duration_since(origin).as_micros() }
    }

    pub fn section(&self) -> TimingSection {
        self.section
    }

    pub fn timestamp(&self) -> u128 {
        self.timestamp
    }
}

/// Manages emission of start/end section timings, enabled through `--json=timings`.
pub struct TimingSectionHandler {
    /// Time when the compilation session started.
    /// If `None`, timing is disabled.
    origin: Option<Instant>,
}

impl TimingSectionHandler {
    pub fn new(enabled: bool) -> Self {
        let origin = if enabled { Some(Instant::now()) } else { None };
        Self { origin }
    }

    /// Returns a RAII guard that will immediately emit a start the provided section, and then emit
    /// its end when it is dropped.
    pub fn start_section<'a>(
        &self,
        diag_ctxt: DiagCtxtHandle<'a>,
        section: TimingSection,
    ) -> TimingSectionGuard<'a> {
        TimingSectionGuard::create(diag_ctxt, section, self.origin)
    }
}

/// RAII wrapper for starting and ending section timings.
pub struct TimingSectionGuard<'a> {
    dcx: DiagCtxtHandle<'a>,
    section: TimingSection,
    origin: Option<Instant>,
}

impl<'a> TimingSectionGuard<'a> {
    fn create(dcx: DiagCtxtHandle<'a>, section: TimingSection, origin: Option<Instant>) -> Self {
        if let Some(origin) = origin {
            dcx.emit_timing_section_start(TimingRecord::from_origin(origin, section));
        }
        Self { dcx, section, origin }
    }
}

impl<'a> Drop for TimingSectionGuard<'a> {
    fn drop(&mut self) {
        if let Some(origin) = self.origin {
            self.dcx.emit_timing_section_end(TimingRecord::from_origin(origin, self.section));
        }
    }
}
