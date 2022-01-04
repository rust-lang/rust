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

// pub(crate) so tests can use the TokenStream, more notes in test/utils.rs
pub(crate) mod abi_1_47;
mod abi_1_54;
mod abi_1_56;
mod abi_1_58;

use super::dylib::LoadProcMacroDylibError;
pub(crate) use abi_1_47::Abi as Abi_1_47;
pub(crate) use abi_1_54::Abi as Abi_1_54;
pub(crate) use abi_1_56::Abi as Abi_1_56;
pub(crate) use abi_1_58::Abi as Abi_1_58;
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
    Abi1_47(Abi_1_47),
    Abi1_54(Abi_1_54),
    Abi1_56(Abi_1_56),
    Abi1_58(Abi_1_58),
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
            (1, 47..=53) => {
                let inner = unsafe { Abi_1_47::from_lib(lib, symbol_name) }?;
                Ok(Abi::Abi1_47(inner))
            }
            (1, 54..=55) => {
                let inner = unsafe { Abi_1_54::from_lib(lib, symbol_name) }?;
                Ok(Abi::Abi1_54(inner))
            }
            (1, 56..=57) => {
                let inner = unsafe { Abi_1_56::from_lib(lib, symbol_name) }?;
                Ok(Abi::Abi1_56(inner))
            }
            (1, 58..) => {
                let inner = unsafe { Abi_1_58::from_lib(lib, symbol_name) }?;
                Ok(Abi::Abi1_58(inner))
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
            Self::Abi1_47(abi) => abi.expand(macro_name, macro_body, attributes),
            Self::Abi1_54(abi) => abi.expand(macro_name, macro_body, attributes),
            Self::Abi1_56(abi) => abi.expand(macro_name, macro_body, attributes),
            Self::Abi1_58(abi) => abi.expand(macro_name, macro_body, attributes),
        }
    }

    pub fn list_macros(&self) -> Vec<(String, ProcMacroKind)> {
        match self {
            Self::Abi1_47(abi) => abi.list_macros(),
            Self::Abi1_54(abi) => abi.list_macros(),
            Self::Abi1_56(abi) => abi.list_macros(),
            Self::Abi1_58(abi) => abi.list_macros(),
        }
    }
}
