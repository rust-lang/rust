use std::time::Instant;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lock;

use crate::DiagCtxtHandle;

/// A high-level section of the compilation process.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TimingSection {
    /// Time spent doing codegen.
    Codegen,
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
    /// Sanity check to ensure that we open and close sections correctly.
    opened_sections: Lock<FxHashSet<TimingSection>>,
}

impl TimingSectionHandler {
    pub fn new(enabled: bool) -> Self {
        let origin = if enabled { Some(Instant::now()) } else { None };
        Self { origin, opened_sections: Lock::new(FxHashSet::default()) }
    }

    /// Returns a RAII guard that will immediately emit a start the provided section, and then emit
    /// its end when it is dropped.
    pub fn section_guard<'a>(
        &self,
        diag_ctxt: DiagCtxtHandle<'a>,
        section: TimingSection,
    ) -> TimingSectionGuard<'a> {
        if self.is_enabled() && self.opened_sections.borrow().contains(&section) {
            diag_ctxt
                .bug(format!("Section `{section:?}` was started again before it was finished"));
        }

        TimingSectionGuard::create(diag_ctxt, section, self.origin)
    }

    /// Start the provided section.
    pub fn start_section(&self, diag_ctxt: DiagCtxtHandle<'_>, section: TimingSection) {
        if let Some(origin) = self.origin {
            let mut opened = self.opened_sections.borrow_mut();
            if !opened.insert(section) {
                diag_ctxt
                    .bug(format!("Section `{section:?}` was started again before it was finished"));
            }

            diag_ctxt.emit_timing_section_start(TimingRecord::from_origin(origin, section));
        }
    }

    /// End the provided section.
    pub fn end_section(&self, diag_ctxt: DiagCtxtHandle<'_>, section: TimingSection) {
        if let Some(origin) = self.origin {
            let mut opened = self.opened_sections.borrow_mut();
            if !opened.remove(&section) {
                diag_ctxt.bug(format!("Section `{section:?}` was ended before being started"));
            }

            diag_ctxt.emit_timing_section_end(TimingRecord::from_origin(origin, section));
        }
    }

    fn is_enabled(&self) -> bool {
        self.origin.is_some()
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
