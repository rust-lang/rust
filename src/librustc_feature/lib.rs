//! # Feature gating
//!
//! This module implements the gating necessary for preventing certain compiler
//! features from being used by default. This module will crawl a pre-expanded
//! AST to ensure that there are no features which are used that are not
//! enabled.
//!
//! Features are enabled in programs via the crate-level attributes of
//! `#![feature(...)]` with a comma-separated list of features.
//!
//! For the purpose of future feature-tracking, once code for detection of feature
//! gate usage is added, *do not remove it again* even once the feature
//! becomes stable.

mod accepted;
mod removed;
mod active;

use std::fmt;
use std::num::NonZeroU32;
use syntax_pos::{Span, edition::Edition, symbol::Symbol};

#[derive(Clone, Copy)]
pub enum State {
    Accepted,
    Active { set: fn(&mut Features, Span) },
    Removed { reason: Option<&'static str> },
    Stabilized { reason: Option<&'static str> },
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            State::Accepted { .. } => write!(f, "accepted"),
            State::Active { .. } => write!(f, "active"),
            State::Removed { .. } => write!(f, "removed"),
            State::Stabilized { .. } => write!(f, "stabilized"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Feature {
    pub state: State,
    pub name: Symbol,
    pub since: &'static str,
    issue: Option<u32>,  // FIXME: once #58732 is done make this an Option<NonZeroU32>
    pub edition: Option<Edition>,
    description: &'static str,
}

impl Feature {
    // FIXME(Centril): privatize again.
    pub fn issue(&self) -> Option<NonZeroU32> {
        self.issue.and_then(|i| NonZeroU32::new(i))
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Stability {
    Unstable,
    // First argument is tracking issue link; second argument is an optional
    // help message, which defaults to "remove this attribute".
    Deprecated(&'static str, Option<&'static str>),
}

pub use accepted::ACCEPTED_FEATURES;
pub use active::{ACTIVE_FEATURES, Features, INCOMPLETE_FEATURES};
pub use removed::{REMOVED_FEATURES, STABLE_REMOVED_FEATURES};
