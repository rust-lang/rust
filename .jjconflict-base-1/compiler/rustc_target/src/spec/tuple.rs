use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::{fmt, io};

use rustc_error_messages::into_diag_arg_using_display;
use rustc_fs_util::try_canonicalize;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

/// Either a target tuple string or a path to a JSON file.
#[derive(Clone, Debug)]
pub enum TargetTuple {
    TargetTuple(String),
    TargetJson {
        /// Warning: This field may only be used by rustdoc. Using it anywhere else will lead to
        /// inconsistencies as it is discarded during serialization.
        path_for_rustdoc: PathBuf,
        tuple: String,
        contents: String,
    },
}

// Use a manual implementation to ignore the path field
impl PartialEq for TargetTuple {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::TargetTuple(l0), Self::TargetTuple(r0)) => l0 == r0,
            (
                Self::TargetJson { path_for_rustdoc: _, tuple: l_tuple, contents: l_contents },
                Self::TargetJson { path_for_rustdoc: _, tuple: r_tuple, contents: r_contents },
            ) => l_tuple == r_tuple && l_contents == r_contents,
            _ => false,
        }
    }
}

// Use a manual implementation to ignore the path field
impl Hash for TargetTuple {
    fn hash<H: Hasher>(&self, state: &mut H) -> () {
        match self {
            TargetTuple::TargetTuple(tuple) => {
                0u8.hash(state);
                tuple.hash(state)
            }
            TargetTuple::TargetJson { path_for_rustdoc: _, tuple, contents } => {
                1u8.hash(state);
                tuple.hash(state);
                contents.hash(state)
            }
        }
    }
}

// Use a manual implementation to prevent encoding the target json file path in the crate metadata
impl<S: Encoder> Encodable<S> for TargetTuple {
    fn encode(&self, s: &mut S) {
        match self {
            TargetTuple::TargetTuple(tuple) => {
                s.emit_u8(0);
                s.emit_str(tuple);
            }
            TargetTuple::TargetJson { path_for_rustdoc: _, tuple, contents } => {
                s.emit_u8(1);
                s.emit_str(tuple);
                s.emit_str(contents);
            }
        }
    }
}

impl<D: Decoder> Decodable<D> for TargetTuple {
    fn decode(d: &mut D) -> Self {
        match d.read_u8() {
            0 => TargetTuple::TargetTuple(d.read_str().to_owned()),
            1 => TargetTuple::TargetJson {
                path_for_rustdoc: PathBuf::new(),
                tuple: d.read_str().to_owned(),
                contents: d.read_str().to_owned(),
            },
            _ => {
                panic!("invalid enum variant tag while decoding `TargetTuple`, expected 0..2");
            }
        }
    }
}

impl TargetTuple {
    /// Creates a target tuple from the passed target tuple string.
    pub fn from_tuple(tuple: &str) -> Self {
        TargetTuple::TargetTuple(tuple.into())
    }

    /// Creates a target tuple from the passed target path.
    pub fn from_path(path: &Path) -> Result<Self, io::Error> {
        let canonicalized_path = try_canonicalize(path)?;
        let contents = std::fs::read_to_string(&canonicalized_path).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("target path {canonicalized_path:?} is not a valid file: {err}"),
            )
        })?;
        let tuple = canonicalized_path
            .file_stem()
            .expect("target path must not be empty")
            .to_str()
            .expect("target path must be valid unicode")
            .to_owned();
        Ok(TargetTuple::TargetJson { path_for_rustdoc: canonicalized_path, tuple, contents })
    }

    /// Returns a string tuple for this target.
    ///
    /// If this target is a path, the file name (without extension) is returned.
    pub fn tuple(&self) -> &str {
        match *self {
            TargetTuple::TargetTuple(ref tuple) | TargetTuple::TargetJson { ref tuple, .. } => {
                tuple
            }
        }
    }

    /// Returns an extended string tuple for this target.
    ///
    /// If this target is a path, a hash of the path is appended to the tuple returned
    /// by `tuple()`.
    pub fn debug_tuple(&self) -> String {
        use std::hash::DefaultHasher;

        match self {
            TargetTuple::TargetTuple(tuple) => tuple.to_owned(),
            TargetTuple::TargetJson { path_for_rustdoc: _, tuple, contents: content } => {
                let mut hasher = DefaultHasher::new();
                content.hash(&mut hasher);
                let hash = hasher.finish();
                format!("{tuple}-{hash}")
            }
        }
    }
}

impl fmt::Display for TargetTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.debug_tuple())
    }
}

into_diag_arg_using_display!(&TargetTuple);
