//! Procedural macros are implemented by compiling the macro providing crate
//! to a dynamic library with a particular ABI which the compiler uses to expand
//! macros. Unfortunately this ABI is not specified and can change from version
//! to version of the compiler. To support this we copy the ABI from the rust
//! compiler into submodules of this module (e.g proc_macro_srv::abis::abi_1_47).
//!
//! All of these ABIs are subsumed in the `Abi` enum, which exposes a simple
//! interface the rest of rust analyzer can use to talk to the macro
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

mod abi_1_58;
mod abi_1_63;
mod abi_1_64;

// Used by `test/utils.rs`
#[cfg(test)]
pub(crate) use abi_1_64::TokenStream as TestTokenStream;

use super::dylib::LoadProcMacroDylibError;
pub(crate) use abi_1_58::Abi as Abi_1_58;
pub(crate) use abi_1_63::Abi as Abi_1_63;
pub(crate) use abi_1_64::Abi as Abi_1_64;
use libloading::Library;
use proc_macro_api::{ProcMacroKind, RustCInfo};

pub struct PanicMessage {
    message: Option<String>,
}

impl PanicMessage {
    pub fn as_str(&self) -> Option<String> {
        self.message.clone()
    }
}

pub(crate) enum Abi {
    Abi1_58(Abi_1_58),
    Abi1_63(Abi_1_63),
    Abi1_64(Abi_1_64),
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
        // FIXME: this should use exclusive ranges when they're stable
        // https://github.com/rust-lang/rust/issues/37854
        match (info.version.0, info.version.1) {
            (1, 58..=62) => {
                let inner = unsafe { Abi_1_58::from_lib(lib, symbol_name) }?;
                Ok(Abi::Abi1_58(inner))
            }
            (1, 63) => {
                let inner = unsafe { Abi_1_63::from_lib(lib, symbol_name) }?;
                Ok(Abi::Abi1_63(inner))
            }
            (1, 64..) => {
                let inner = unsafe { Abi_1_64::from_lib(lib, symbol_name) }?;
                Ok(Abi::Abi1_64(inner))
            }
            _ => Err(LoadProcMacroDylibError::UnsupportedABI),
        }
    }

    pub fn expand(
        &self,
        macro_name: &str,
        macro_body: &tt::Subtree,
        attributes: Option<&tt::Subtree>,
    ) -> Result<tt::Subtree, PanicMessage> {
        match self {
            Self::Abi1_58(abi) => abi.expand(macro_name, macro_body, attributes),
            Self::Abi1_63(abi) => abi.expand(macro_name, macro_body, attributes),
            Self::Abi1_64(abi) => abi.expand(macro_name, macro_body, attributes),
        }
    }

    pub fn list_macros(&self) -> Vec<(String, ProcMacroKind)> {
        match self {
            Self::Abi1_58(abi) => abi.list_macros(),
            Self::Abi1_63(abi) => abi.list_macros(),
            Self::Abi1_64(abi) => abi.list_macros(),
        }
    }
}
