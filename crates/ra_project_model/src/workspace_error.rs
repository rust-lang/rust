//! Workspace-related errors

use std::{error::Error, fmt, io, path::PathBuf, string::FromUtf8Error};

use ra_db::ParseEditionError;

#[derive(Debug)]
pub enum WorkspaceError {
    CargoMetadataFailed(cargo_metadata::Error),
    CargoTomlNotFound(PathBuf),
    NoStdLib(PathBuf),
    OpenWorkspaceError(io::Error),
    ParseEditionError(ParseEditionError),
    ReadWorkspaceError(serde_json::Error),
    RustcCfgError,
    RustcError(io::Error),
    RustcOutputError(FromUtf8Error),
    SysrootNotFound,
}

impl fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenWorkspaceError(err) | Self::RustcError(err) => write!(f, "{}", err),
            Self::ParseEditionError(err) => write!(f, "{}", err),
            Self::ReadWorkspaceError(err) => write!(f, "{}", err),
            Self::RustcOutputError(err) => write!(f, "{}", err),
            Self::CargoMetadataFailed(err) => write!(f, "cargo metadata failed: {}", err),
            Self::RustcCfgError => write!(f, "failed to get rustc cfgs"),
            Self::SysrootNotFound => write!(f, "failed to locate sysroot"),
            Self::CargoTomlNotFound(path) => {
                write!(f, "can't find Cargo.toml at {}", path.display())
            }
            Self::NoStdLib(sysroot) => write!(
                f,
                "can't load standard library from sysroot\n\
                 {:?}\n\
                 try running `rustup component add rust-src` or set `RUST_SRC_PATH`",
                sysroot,
            ),
        }
    }
}

impl From<ParseEditionError> for WorkspaceError {
    fn from(err: ParseEditionError) -> Self {
        Self::ParseEditionError(err.into())
    }
}

impl From<cargo_metadata::Error> for WorkspaceError {
    fn from(err: cargo_metadata::Error) -> Self {
        Self::CargoMetadataFailed(err.into())
    }
}

impl Error for WorkspaceError {}
