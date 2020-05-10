//! General-purpose instrumentation for progress reporting.
//!
//! Note:
//! Most of the methods accept `&mut self` just to be more restrictive (for forward compat)
//! even tho for some of them we can weaken this requirement to shared reference (`&self`).

use crossbeam_channel::Receiver;
use std::fmt;

#[derive(Debug)]
pub enum ProgressStatus<B, P> {
    Begin(B),
    Progress(P),
    End,
}

pub struct Progress<B, P>(Option<crossbeam_channel::Sender<ProgressStatus<B, P>>>);
impl<B, P> Progress<B, P> {
    pub fn report(&mut self, payload: P) {
        self.report_with(|| payload);
    }

    pub fn report_with(&mut self, payload: impl FnOnce() -> P) {
        self.send_status(|| ProgressStatus::Progress(payload()));
    }

    fn send_status(&self, status: impl FnOnce() -> ProgressStatus<B, P>) {
        if let Some(sender) = &self.0 {
            sender.try_send(status()).expect("progress report must not block");
        }
    }
}

impl<B, P> Drop for Progress<B, P> {
    fn drop(&mut self) {
        self.send_status(|| ProgressStatus::End);
    }
}

pub struct ProgressSource<B, P>(Option<crossbeam_channel::Sender<ProgressStatus<B, P>>>);
impl<B, P> ProgressSource<B, P> {
    pub fn real_if(real: bool) -> (Receiver<ProgressStatus<B, P>>, Self) {
        if real {
            let (sender, receiver) = crossbeam_channel::unbounded();
            (receiver, Self(Some(sender)))
        } else {
            (crossbeam_channel::never(), Self(None))
        }
    }

    pub fn begin(&mut self, payload: B) -> Progress<B, P> {
        self.begin_with(|| payload)
    }

    pub fn begin_with(&mut self, payload: impl FnOnce() -> B) -> Progress<B, P> {
        let progress = Progress(self.0.clone());
        progress.send_status(|| ProgressStatus::Begin(payload()));
        progress
    }
}

impl<B, P> Clone for ProgressSource<B, P> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<B, P> fmt::Debug for ProgressSource<B, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ProgressSource").field(&self.0).finish()
    }
}

pub type U32ProgressStatus = ProgressStatus<U32ProgressReport, U32ProgressReport>;

#[derive(Debug)]
pub struct U32ProgressReport {
    pub processed: u32,
    pub total: u32,
}
impl U32ProgressReport {
    pub fn percentage(&self) -> f64 {
        f64::from(100 * self.processed) / f64::from(self.total)
    }
    pub fn to_message(&self, prefix: &str, unit: &str) -> String {
        format!("{} ({}/{} {})", prefix, self.processed, self.total, unit)
    }
}

pub struct U32Progress {
    inner: Progress<U32ProgressReport, U32ProgressReport>,
    processed: u32,
    total: u32,
}

#[derive(Debug, Eq, PartialEq)]
pub struct IsDone(pub bool);

impl U32Progress {
    pub fn report(&mut self, new_processed: u32) -> IsDone {
        if self.processed < new_processed {
            self.processed = new_processed;
            self.inner.report(U32ProgressReport { processed: new_processed, total: self.total });
        }
        IsDone(self.processed >= self.total)
    }
}

#[derive(Clone)]
pub struct U32ProgressSource {
    inner: ProgressSource<U32ProgressReport, U32ProgressReport>,
}

impl U32ProgressSource {
    pub fn real_if(
        real: bool,
    ) -> (Receiver<ProgressStatus<U32ProgressReport, U32ProgressReport>>, Self) {
        let (recv, inner) = ProgressSource::real_if(real);
        (recv, Self { inner })
    }

    pub fn begin(&mut self, initial: u32, total: u32) -> U32Progress {
        U32Progress {
            inner: self.inner.begin(U32ProgressReport { processed: initial, total }),
            processed: initial,
            total,
        }
    }
}
