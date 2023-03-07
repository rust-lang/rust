//! Procedural macros are implemented by compiling the macro providing crate
//! to a dynamic library with a particular ABI which the compiler uses to expand
//! macros. Unfortunately this ABI is not specified and can change from version
//! to version of the compiler. To support this we copy the ABI from the rust
//! compiler into submodules of this module (e.g proc_macro_srv::abis::abi_1_47).
//!
//! All of these ABIs are subsumed in the `Abi` enum, which exposes a simple
//! interface the rest of rust-analyzer can use to talk to the macro
//! provider.
//!
//! # Adding a new ABI
//!
//! To add a new ABI you'll need to copy the source of the target proc_macro
//! crate from the source tree of the Rust compiler into this directory tree.
//! Then you'll need to modify it
//! - Remove any feature! or other things which won't compile on stable
//! - change any absolute imports to relative imports within the ABI tree
//!
//! Then you'll need to add a branch to the `Abi` enum and an implementation of
//! `Abi::expand`, `Abi::list_macros` and `Abi::from_lib` for the new ABI. See
//! `proc_macro_srv/src/abis/abi_1_47/mod.rs` for an example. Finally you'll
//! need to update the conditionals in `Abi::from_lib` to return your new ABI
//! for the relevant versions of the rust compiler
//!

mod abi_1_63;
#[cfg(feature = "sysroot-abi")]
mod abi_sysroot;

// see `build.rs`
include!(concat!(env!("OUT_DIR"), "/rustc_version.rs"));

// Used by `test/utils.rs`
#[cfg(all(test, feature = "sysroot-abi"))]
pub(crate) use abi_sysroot::TokenStream as TestTokenStream;

use super::dylib::LoadProcMacroDylibError;
pub(crate) use abi_1_63::Abi as Abi_1_63;
#[cfg(feature = "sysroot-abi")]
pub(crate) use abi_sysroot::Abi as Abi_Sysroot;
use libloading::Library;
use proc_macro_api::{ProcMacroKind, RustCInfo};

use crate::tt;

pub struct PanicMessage {
    message: Option<String>,
}

impl PanicMessage {
    pub fn as_str(&self) -> Option<String> {
        self.message.clone()
    }
}

pub(crate) enum Abi {
    Abi1_63(Abi_1_63),
    #[cfg(feature = "sysroot-abi")]
    AbiSysroot(Abi_Sysroot),
}

impl Abi {
    /// Load a new ABI.
    ///
    /// # Arguments
    ///
    /// *`lib` - The dynamic library containing the macro implementations
    /// *`symbol_name` - The symbol name the macros can be found attributes
    /// *`info` - RustCInfo about the compiler that was used to compile the
    ///           macro crate. This is the information we use to figure out
    ///           which ABI to return
    pub fn from_lib(
        lib: &Library,
        symbol_name: String,
        info: RustCInfo,
    ) -> Result<Abi, LoadProcMacroDylibError> {
        // the sysroot ABI relies on `extern proc_macro` with unstable features,
        // instead of a snapshot of the proc macro bridge's source code. it's only
        // enabled if we have an exact version match.
        #[cfg(feature = "sysroot-abi")]
        {
            if info.version_string == RUSTC_VERSION_STRING {
                let inner = unsafe { Abi_Sysroot::from_lib(lib, symbol_name) }?;
                return Ok(Abi::AbiSysroot(inner));
            }

            // if we reached this point, versions didn't match. in testing, we
            // want that to panic - this could mean that the format of `rustc
            // --version` no longer matches the format of the version string
            // stored in the `.rustc` section, and we want to catch that in-tree
            // with `x.py test`
            #[cfg(test)]
            {
                let allow_mismatch = std::env::var("PROC_MACRO_SRV_ALLOW_SYSROOT_MISMATCH");
                if let Ok("1") = allow_mismatch.as_deref() {
                    // only used by rust-analyzer developers, when working on the
                    // sysroot ABI from the rust-analyzer repository - which should
                    // only happen pre-subtree. this can be removed later.
                } else {
                    panic!(
                        "sysroot ABI mismatch: dylib rustc version (read from .rustc section): {:?} != proc-macro-srv version (read from 'rustc --version'): {:?}",
                        info.version_string, RUSTC_VERSION_STRING
                    );
                }
            }
        }

        // FIXME: this should use exclusive ranges when they're stable
        // https://github.com/rust-lang/rust/issues/37854
        match (info.version.0, info.version.1) {
            (1, 63) => {
                let inner = unsafe { Abi_1_63::from_lib(lib, symbol_name) }?;
                Ok(Abi::Abi1_63(inner))
            }
            _ => Err(LoadProcMacroDylibError::UnsupportedABI(info.version_string)),
        }
    }

    pub fn expand(
        &self,
        macro_name: &str,
        macro_body: &tt::Subtree,
        attributes: Option<&tt::Subtree>,
    ) -> Result<tt::Subtree, PanicMessage> {
        match self {
            Self::Abi1_63(abi) => abi.expand(macro_name, macro_body, attributes),
            #[cfg(feature = "sysroot-abi")]
            Self::AbiSysroot(abi) => abi.expand(macro_name, macro_body, attributes),
        }
    }

    pub fn list_macros(&self) -> Vec<(String, ProcMacroKind)> {
        match self {
            Self::Abi1_63(abi) => abi.list_macros(),
            #[cfg(feature = "sysroot-abi")]
            Self::AbiSysroot(abi) => abi.list_macros(),
        }
    }
}

#[test]
fn test_version_check() {
    let path = paths::AbsPathBuf::assert(crate::proc_macro_test_dylib_path());
    let info = proc_macro_api::read_dylib_info(&path).unwrap();
    assert!(info.version.1 >= 50);
}
