use crate::DiagCtxtHandle;

/// A high-level section of the compilation process.
#[derive(Copy, Clone, Debug)]
pub enum TimingSection {
    /// Time spent linking.
    Linking,
}

/// Manages emission of start/end section timings, enabled through `--json=timings`.
pub struct TimingSectionHandler {
    enabled: bool,
}

impl TimingSectionHandler {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Returns a RAII guard that will immediately emit a start the provided section, and then emit
    /// its end when it is dropped.
    pub fn start_section<'a>(
        &self,
        diag_ctxt: DiagCtxtHandle<'a>,
        section: TimingSection,
    ) -> TimingSectionGuard<'a> {
        TimingSectionGuard::create(diag_ctxt, section, self.enabled)
    }
}

pub struct TimingSectionGuard<'a> {
    dcx: DiagCtxtHandle<'a>,
    section: TimingSection,
    enabled: bool,
}

impl<'a> TimingSectionGuard<'a> {
    fn create(dcx: DiagCtxtHandle<'a>, section: TimingSection, enabled: bool) -> Self {
        if enabled {
            dcx.emit_timing_section_start(section);
        }
        Self { dcx, section, enabled }
    }
}

impl<'a> Drop for TimingSectionGuard<'a> {
    fn drop(&mut self) {
        if self.enabled {
            self.dcx.emit_timing_section_end(self.section);
        }
    }
}
