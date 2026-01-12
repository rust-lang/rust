use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use crate::expand::{Decodable, Encodable, HashStable_Generic};

#[derive(Clone, Copy, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum BatchMode {
    Error,
    Source,
}

#[derive(Clone, Copy, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum BatchActivity {
    Const,
    Leaf,
    Vector,
    FakeActivitySize,
}

/// We generate one of these structs for each variable inside the `#[batch(...)]` attribute.
#[derive(Clone, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct BatchItem {
    /// The name of the function getting differentiated
    pub source: String,
    /// The name of the function being generated
    pub target: String,
    pub attrs: BatchAttrs,
}
#[derive(Clone, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct BatchAttrs {
    pub mode: BatchMode,
    pub width: usize,
    pub input_activity: Vec<BatchActivity>,
}

impl Display for BatchMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            BatchMode::Error => write!(f, "Error"),
            BatchMode::Source => write!(f, "Source"),
        }
    }
}

impl Display for BatchActivity {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchActivity::Const => write!(f, "Const"),
            BatchActivity::Leaf => write!(f, "Leaf"),
            BatchActivity::Vector => write!(f, "Vector"),
            BatchActivity::FakeActivitySize => write!(f, "FakeActivitySize"),
        }
    }
}

impl FromStr for BatchMode {
    type Err = ();

    fn from_str(s: &str) -> Result<BatchMode, ()> {
        match s {
            "Error" => Ok(BatchMode::Error),
            "Source" => Ok(BatchMode::Source),
            _ => Err(()),
        }
    }
}

impl FromStr for BatchActivity {
    type Err = ();

    fn from_str(s: &str) -> Result<BatchActivity, ()> {
        match s {
            "Const" => Ok(BatchActivity::Const),
            "Leaf" => Ok(BatchActivity::Leaf),
            "Vector" => Ok(BatchActivity::Vector),
            _ => Err(()),
        }
    }
}
