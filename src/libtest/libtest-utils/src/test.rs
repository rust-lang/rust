//! Tests utilities

use crate::{format::Padding, ShouldPanic};
use std::{borrow::Cow, fmt};

/// Test description.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Desc {
    pub name: Name,
    pub ignore: bool,
    pub should_panic: ShouldPanic,
    pub allow_fail: bool,
}

impl Desc {
    /// Padded test name
    pub fn padded_name(&self, column_count: usize, align: Padding) -> String {
        let mut name = String::from(self.name.as_slice());
        let fill = column_count.saturating_sub(name.len());
        let pad = " ".repeat(fill);
        match align {
            Padding::None => name,
            Padding::OnRight => {
                name.push_str(&pad);
                name
            }
        }
    }
}

/// The name of a test.
///
/// By convention this follows the rules for rust paths; i.e., it should be a
/// series of identifiers separated by double colons. This way if some test
/// runner wants to arrange the tests hierarchically it may.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Name {
    Static(&'static str),
    Dyn(String),
    Aligned(Cow<'static, str>, Padding),
}

impl Name {
    pub fn as_slice(&self) -> &str {
        match *self {
            Name::Static(s) => s,
            Name::Dyn(ref s) => s,
            Name::Aligned(ref s, _) => &*s,
        }
    }

    pub fn padding(&self) -> Padding {
        match self {
            &Name::Aligned(_, p) => p,
            _ => Padding::None,
        }
    }

    pub fn with_padding(&self, padding: Padding) -> Name {
        let name = match self {
            &Name::Static(name) => Cow::Borrowed(name),
            &Name::Dyn(ref name) => Cow::Owned(name.clone()),
            &Name::Aligned(ref name, _) => name.clone(),
        };

        Name::Aligned(name, padding)
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.as_slice(), f)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Result {
    Ok,
    Failed,
    FailedMsg(String),
    Ignored,
    AllowedFail,
}

unsafe impl Send for Result {}
