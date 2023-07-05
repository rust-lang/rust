//! [Flexible target specification.](https://github.com/rust-lang/rfcs/pull/131)
//!
//! Rust targets a wide variety of usecases, and in the interest of flexibility,
//! allows new target triples to be defined in configuration files. Most users
//! will not need to care about these, but this is invaluable when porting Rust
//! to a new platform, and allows for an unprecedented level of control over how
//! the compiler works.
//!
//! # Using custom targets
//!
//! A target triple, as passed via `rustc --target=TRIPLE`, will first be
//! compared against the list of built-in targets. This is to ease distributing
//! rustc (no need for configuration files) and also to hold these built-in
//! targets as immutable and sacred. If `TRIPLE` is not one of the built-in
//! targets, rustc will check if a file named `TRIPLE` exists. If it does, it
//! will be loaded as the target configuration. If the file does not exist,
//! rustc will search each directory in the environment variable
//! `RUST_TARGET_PATH` for a file named `TRIPLE.json`. The first one found will
//! be loaded. If no file is found in any of those directories, a fatal error
//! will be given.
//!
//! Projects defining their own targets should use
//! `--target=path/to/my-awesome-platform.json` instead of adding to
//! `RUST_TARGET_PATH`.
//!
//! # Defining a new target
//!
//! Targets are defined using [JSON](https://json.org/). The `Target` struct in
//! this module defines the format the JSON file should take, though each
//! underscore in the field names should be replaced with a hyphen (`-`) in the
//! JSON file. Some fields are required in every target specification, such as
//! `llvm-target`, `target-endian`, `target-pointer-width`, `data-layout`,
//! `arch`, and `os`. In general, options passed to rustc with `-C` override
//! the target's settings, though `target-feature` and `link-args` will *add*
//! to the list specified by the target, rather than replace.

use crate::abi::call::Conv;
use crate::abi::{Endian, Integer, Size, TargetDataLayout, TargetDataLayoutErrors};
use crate::json::{Json, ToJson};
use crate::spec::abi::{lookup as lookup_abi, Abi};
use crate::spec::crt_objects::{CrtObjects, LinkSelfContainedDefault};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_fs_util::try_canonicalize;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_span::symbol::{sym, Symbol};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{fmt, io};

use rustc_macros::HashStable_Generic;

pub mod abi;
pub mod crt_objects;

mod aix_base;
mod android_base;
mod apple_base;
pub use apple_base::deployment_target as current_apple_deployment_target;
mod avr_gnu_base;
pub use avr_gnu_base::ef_avr_arch;
mod bpf_base;
mod dragonfly_base;
mod freebsd_base;
mod fuchsia_base;
mod haiku_base;
mod hermit_base;
mod illumos_base;
mod l4re_base;
mod linux_base;
mod linux_gnu_base;
mod linux_musl_base;
mod linux_uclibc_base;
mod msvc_base;
mod netbsd_base;
mod nto_qnx_base;
mod openbsd_base;
mod redox_base;
mod solaris_base;
mod solid_base;
mod thumb_base;
mod uefi_msvc_base;
mod vxworks_base;
mod wasm_base;
mod windows_gnu_base;
mod windows_gnullvm_base;
mod windows_msvc_base;
mod windows_uwp_gnu_base;
mod windows_uwp_msvc_base;

/// Linker is called through a C/C++ compiler.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Cc {
    Yes,
    No,
}

/// Linker is LLD.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Lld {
    Yes,
    No,
}

/// All linkers have some kinds of command line interfaces and rustc needs to know which commands
/// to use with each of them. So we cluster all such interfaces into a (somewhat arbitrary) number
/// of classes that we call "linker flavors".
///
/// Technically, it's not even necessary, we can nearly always infer the flavor from linker name
/// and target properties like `is_like_windows`/`is_like_osx`/etc. However, the PRs originally
/// introducing `-Clinker-flavor` (#40018 and friends) were aiming to reduce this kind of inference
/// and provide something certain and explicitly specified instead, and that design goal is still
/// relevant now.
///
/// The second goal is to keep the number of flavors to the minimum if possible.
/// LLD somewhat forces our hand here because that linker is self-sufficient only if its executable
/// (`argv[0]`) is named in specific way, otherwise it doesn't work and requires a
/// `-flavor LLD_FLAVOR` argument to choose which logic to use. Our shipped `rust-lld` in
/// particular is not named in such specific way, so it needs the flavor option, so we make our
/// linker flavors sufficiently fine-grained to satisfy LLD without inferring its flavor from other
/// target properties, in accordance with the first design goal.
///
/// The first component of the flavor is tightly coupled with the compilation target,
/// while the `Cc` and `Lld` flags can vary within the same target.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LinkerFlavor {
    /// Unix-like linker with GNU extensions (both naked and compiler-wrapped forms).
    /// Besides similar "default" Linux/BSD linkers this also includes Windows/GNU linker,
    /// which is somewhat different because it doesn't produce ELFs.
    Gnu(Cc, Lld),
    /// Unix-like linker for Apple targets (both naked and compiler-wrapped forms).
    /// Extracted from the "umbrella" `Unix` flavor due to its corresponding LLD flavor.
    Darwin(Cc, Lld),
    /// Unix-like linker for Wasm targets (both naked and compiler-wrapped forms).
    /// Extracted from the "umbrella" `Unix` flavor due to its corresponding LLD flavor.
    /// Non-LLD version does not exist, so the lld flag is currently hardcoded here.
    WasmLld(Cc),
    /// Basic Unix-like linker for "any other Unix" targets (Solaris/illumos, L4Re, MSP430, etc),
    /// possibly with non-GNU extensions (both naked and compiler-wrapped forms).
    /// LLD doesn't support any of these.
    Unix(Cc),
    /// MSVC-style linker for Windows and UEFI, LLD supports it.
    Msvc(Lld),
    /// Emscripten Compiler Frontend, a wrapper around `WasmLld(Cc::Yes)` that has a different
    /// interface and produces some additional JavaScript output.
    EmCc,
    // Below: other linker-like tools with unique interfaces for exotic targets.
    /// Linker tool for BPF.
    Bpf,
    /// Linker tool for Nvidia PTX.
    Ptx,
}

/// Linker flavors available externally through command line (`-Clinker-flavor`)
/// or json target specifications.
/// FIXME: This set has accumulated historically, bring it more in line with the internal
/// linker flavors (`LinkerFlavor`).
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LinkerFlavorCli {
    // New (unstable) flavors, with direct counterparts in `LinkerFlavor`.
    Gnu(Cc, Lld),
    Darwin(Cc, Lld),
    WasmLld(Cc),
    Unix(Cc),
    // Note: `Msvc(Lld::No)` is also a stable value.
    Msvc(Lld),
    EmCc,
    Bpf,
    Ptx,

    // Below: the legacy stable values.
    Gcc,
    Ld,
    Lld(LldFlavor),
    Em,
    BpfLinker,
    PtxLinker,
}

impl LinkerFlavorCli {
    /// Returns whether this `-C linker-flavor` option is one of the unstable values.
    pub fn is_unstable(&self) -> bool {
        match self {
            LinkerFlavorCli::Gnu(..)
            | LinkerFlavorCli::Darwin(..)
            | LinkerFlavorCli::WasmLld(..)
            | LinkerFlavorCli::Unix(..)
            | LinkerFlavorCli::Msvc(Lld::Yes)
            | LinkerFlavorCli::EmCc
            | LinkerFlavorCli::Bpf
            | LinkerFlavorCli::Ptx
            | LinkerFlavorCli::BpfLinker
            | LinkerFlavorCli::PtxLinker => true,
            LinkerFlavorCli::Gcc
            | LinkerFlavorCli::Ld
            | LinkerFlavorCli::Lld(..)
            | LinkerFlavorCli::Msvc(Lld::No)
            | LinkerFlavorCli::Em => false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LldFlavor {
    Wasm,
    Ld64,
    Ld,
    Link,
}

impl LldFlavor {
    pub fn as_str(&self) -> &'static str {
        match self {
            LldFlavor::Wasm => "wasm",
            LldFlavor::Ld64 => "darwin",
            LldFlavor::Ld => "gnu",
            LldFlavor::Link => "link",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "darwin" => LldFlavor::Ld64,
            "gnu" => LldFlavor::Ld,
            "link" => LldFlavor::Link,
            "wasm" => LldFlavor::Wasm,
            _ => return None,
        })
    }
}

impl ToJson for LldFlavor {
    fn to_json(&self) -> Json {
        self.as_str().to_json()
    }
}

impl LinkerFlavor {
    /// At this point the target's reference linker flavor doesn't yet exist and we need to infer
    /// it. The inference always succeds and gives some result, and we don't report any flavor
    /// incompatibility errors for json target specs. The CLI flavor is used as the main source
    /// of truth, other flags are used in case of ambiguities.
    fn from_cli_json(cli: LinkerFlavorCli, lld_flavor: LldFlavor, is_gnu: bool) -> LinkerFlavor {
        match cli {
            LinkerFlavorCli::Gnu(cc, lld) => LinkerFlavor::Gnu(cc, lld),
            LinkerFlavorCli::Darwin(cc, lld) => LinkerFlavor::Darwin(cc, lld),
            LinkerFlavorCli::WasmLld(cc) => LinkerFlavor::WasmLld(cc),
            LinkerFlavorCli::Unix(cc) => LinkerFlavor::Unix(cc),
            LinkerFlavorCli::Msvc(lld) => LinkerFlavor::Msvc(lld),
            LinkerFlavorCli::EmCc => LinkerFlavor::EmCc,
            LinkerFlavorCli::Bpf => LinkerFlavor::Bpf,
            LinkerFlavorCli::Ptx => LinkerFlavor::Ptx,

            // Below: legacy stable values
            LinkerFlavorCli::Gcc => match lld_flavor {
                LldFlavor::Ld if is_gnu => LinkerFlavor::Gnu(Cc::Yes, Lld::No),
                LldFlavor::Ld64 => LinkerFlavor::Darwin(Cc::Yes, Lld::No),
                LldFlavor::Wasm => LinkerFlavor::WasmLld(Cc::Yes),
                LldFlavor::Ld | LldFlavor::Link => LinkerFlavor::Unix(Cc::Yes),
            },
            LinkerFlavorCli::Ld => match lld_flavor {
                LldFlavor::Ld if is_gnu => LinkerFlavor::Gnu(Cc::No, Lld::No),
                LldFlavor::Ld64 => LinkerFlavor::Darwin(Cc::No, Lld::No),
                LldFlavor::Ld | LldFlavor::Wasm | LldFlavor::Link => LinkerFlavor::Unix(Cc::No),
            },
            LinkerFlavorCli::Lld(LldFlavor::Ld) => LinkerFlavor::Gnu(Cc::No, Lld::Yes),
            LinkerFlavorCli::Lld(LldFlavor::Ld64) => LinkerFlavor::Darwin(Cc::No, Lld::Yes),
            LinkerFlavorCli::Lld(LldFlavor::Wasm) => LinkerFlavor::WasmLld(Cc::No),
            LinkerFlavorCli::Lld(LldFlavor::Link) => LinkerFlavor::Msvc(Lld::Yes),
            LinkerFlavorCli::Em => LinkerFlavor::EmCc,
            LinkerFlavorCli::BpfLinker => LinkerFlavor::Bpf,
            LinkerFlavorCli::PtxLinker => LinkerFlavor::Ptx,
        }
    }

    fn to_cli(self) -> LinkerFlavorCli {
        match self {
            LinkerFlavor::Gnu(Cc::Yes, _)
            | LinkerFlavor::Darwin(Cc::Yes, _)
            | LinkerFlavor::WasmLld(Cc::Yes)
            | LinkerFlavor::Unix(Cc::Yes) => LinkerFlavorCli::Gcc,
            LinkerFlavor::Gnu(_, Lld::Yes) => LinkerFlavorCli::Lld(LldFlavor::Ld),
            LinkerFlavor::Darwin(_, Lld::Yes) => LinkerFlavorCli::Lld(LldFlavor::Ld64),
            LinkerFlavor::WasmLld(..) => LinkerFlavorCli::Lld(LldFlavor::Wasm),
            LinkerFlavor::Gnu(..) | LinkerFlavor::Darwin(..) | LinkerFlavor::Unix(..) => {
                LinkerFlavorCli::Ld
            }
            LinkerFlavor::Msvc(Lld::Yes) => LinkerFlavorCli::Lld(LldFlavor::Link),
            LinkerFlavor::Msvc(..) => LinkerFlavorCli::Msvc(Lld::No),
            LinkerFlavor::EmCc => LinkerFlavorCli::Em,
            LinkerFlavor::Bpf => LinkerFlavorCli::BpfLinker,
            LinkerFlavor::Ptx => LinkerFlavorCli::PtxLinker,
        }
    }

    fn infer_cli_hints(cli: LinkerFlavorCli) -> (Option<Cc>, Option<Lld>) {
        match cli {
            LinkerFlavorCli::Gnu(cc, lld) | LinkerFlavorCli::Darwin(cc, lld) => {
                (Some(cc), Some(lld))
            }
            LinkerFlavorCli::WasmLld(cc) => (Some(cc), Some(Lld::Yes)),
            LinkerFlavorCli::Unix(cc) => (Some(cc), None),
            LinkerFlavorCli::Msvc(lld) => (Some(Cc::No), Some(lld)),
            LinkerFlavorCli::EmCc => (Some(Cc::Yes), Some(Lld::Yes)),
            LinkerFlavorCli::Bpf | LinkerFlavorCli::Ptx => (None, None),

            // Below: legacy stable values
            LinkerFlavorCli::Gcc => (Some(Cc::Yes), None),
            LinkerFlavorCli::Ld => (Some(Cc::No), Some(Lld::No)),
            LinkerFlavorCli::Lld(_) => (Some(Cc::No), Some(Lld::Yes)),
            LinkerFlavorCli::Em => (Some(Cc::Yes), Some(Lld::Yes)),
            LinkerFlavorCli::BpfLinker | LinkerFlavorCli::PtxLinker => (None, None),
        }
    }

    fn infer_linker_hints(linker_stem: &str) -> (Option<Cc>, Option<Lld>) {
        // Remove any version postfix.
        let stem = linker_stem
            .rsplit_once('-')
            .and_then(|(lhs, rhs)| rhs.chars().all(char::is_numeric).then_some(lhs))
            .unwrap_or(linker_stem);

        // GCC/Clang can have an optional target prefix.
        if stem == "emcc"
            || stem == "gcc"
            || stem.ends_with("-gcc")
            || stem == "g++"
            || stem.ends_with("-g++")
            || stem == "clang"
            || stem.ends_with("-clang")
            || stem == "clang++"
            || stem.ends_with("-clang++")
        {
            (Some(Cc::Yes), None)
        } else if stem == "wasm-ld"
            || stem.ends_with("-wasm-ld")
            || stem == "ld.lld"
            || stem == "lld"
            || stem == "rust-lld"
            || stem == "lld-link"
        {
            (Some(Cc::No), Some(Lld::Yes))
        } else if stem == "ld" || stem.ends_with("-ld") || stem == "link" {
            (Some(Cc::No), Some(Lld::No))
        } else {
            (None, None)
        }
    }

    fn with_hints(self, (cc_hint, lld_hint): (Option<Cc>, Option<Lld>)) -> LinkerFlavor {
        match self {
            LinkerFlavor::Gnu(cc, lld) => {
                LinkerFlavor::Gnu(cc_hint.unwrap_or(cc), lld_hint.unwrap_or(lld))
            }
            LinkerFlavor::Darwin(cc, lld) => {
                LinkerFlavor::Darwin(cc_hint.unwrap_or(cc), lld_hint.unwrap_or(lld))
            }
            LinkerFlavor::WasmLld(cc) => LinkerFlavor::WasmLld(cc_hint.unwrap_or(cc)),
            LinkerFlavor::Unix(cc) => LinkerFlavor::Unix(cc_hint.unwrap_or(cc)),
            LinkerFlavor::Msvc(lld) => LinkerFlavor::Msvc(lld_hint.unwrap_or(lld)),
            LinkerFlavor::EmCc | LinkerFlavor::Bpf | LinkerFlavor::Ptx => self,
        }
    }

    pub fn with_cli_hints(self, cli: LinkerFlavorCli) -> LinkerFlavor {
        self.with_hints(LinkerFlavor::infer_cli_hints(cli))
    }

    pub fn with_linker_hints(self, linker_stem: &str) -> LinkerFlavor {
        self.with_hints(LinkerFlavor::infer_linker_hints(linker_stem))
    }

    pub fn check_compatibility(self, cli: LinkerFlavorCli) -> Option<String> {
        let compatible = |cli| {
            // The CLI flavor should be compatible with the target if:
            // 1. they are counterparts: they have the same principal flavor.
            match (self, cli) {
                (LinkerFlavor::Gnu(..), LinkerFlavorCli::Gnu(..))
                | (LinkerFlavor::Darwin(..), LinkerFlavorCli::Darwin(..))
                | (LinkerFlavor::WasmLld(..), LinkerFlavorCli::WasmLld(..))
                | (LinkerFlavor::Unix(..), LinkerFlavorCli::Unix(..))
                | (LinkerFlavor::Msvc(..), LinkerFlavorCli::Msvc(..))
                | (LinkerFlavor::EmCc, LinkerFlavorCli::EmCc)
                | (LinkerFlavor::Bpf, LinkerFlavorCli::Bpf)
                | (LinkerFlavor::Ptx, LinkerFlavorCli::Ptx) => return true,
                _ => {}
            }

            // 2. or, the flavor is legacy and survives this roundtrip.
            cli == self.with_cli_hints(cli).to_cli()
        };
        (!compatible(cli)).then(|| {
            LinkerFlavorCli::all()
                .iter()
                .filter(|cli| compatible(**cli))
                .map(|cli| cli.desc())
                .intersperse(", ")
                .collect()
        })
    }

    pub fn lld_flavor(self) -> LldFlavor {
        match self {
            LinkerFlavor::Gnu(..)
            | LinkerFlavor::Unix(..)
            | LinkerFlavor::EmCc
            | LinkerFlavor::Bpf
            | LinkerFlavor::Ptx => LldFlavor::Ld,
            LinkerFlavor::Darwin(..) => LldFlavor::Ld64,
            LinkerFlavor::WasmLld(..) => LldFlavor::Wasm,
            LinkerFlavor::Msvc(..) => LldFlavor::Link,
        }
    }

    pub fn is_gnu(self) -> bool {
        matches!(self, LinkerFlavor::Gnu(..))
    }

    /// Returns whether the flavor uses the `lld` linker.
    pub fn uses_lld(self) -> bool {
        // Exhaustive match in case new flavors are added in the future.
        match self {
            LinkerFlavor::Gnu(_, Lld::Yes)
            | LinkerFlavor::Darwin(_, Lld::Yes)
            | LinkerFlavor::WasmLld(..)
            | LinkerFlavor::EmCc
            | LinkerFlavor::Msvc(Lld::Yes) => true,
            LinkerFlavor::Gnu(..)
            | LinkerFlavor::Darwin(..)
            | LinkerFlavor::Msvc(_)
            | LinkerFlavor::Unix(_)
            | LinkerFlavor::Bpf
            | LinkerFlavor::Ptx => false,
        }
    }

    /// Returns whether the flavor calls the linker via a C/C++ compiler.
    pub fn uses_cc(self) -> bool {
        // Exhaustive match in case new flavors are added in the future.
        match self {
            LinkerFlavor::Gnu(Cc::Yes, _)
            | LinkerFlavor::Darwin(Cc::Yes, _)
            | LinkerFlavor::WasmLld(Cc::Yes)
            | LinkerFlavor::Unix(Cc::Yes)
            | LinkerFlavor::EmCc => true,
            LinkerFlavor::Gnu(..)
            | LinkerFlavor::Darwin(..)
            | LinkerFlavor::WasmLld(_)
            | LinkerFlavor::Msvc(_)
            | LinkerFlavor::Unix(_)
            | LinkerFlavor::Bpf
            | LinkerFlavor::Ptx => false,
        }
    }
}

macro_rules! linker_flavor_cli_impls {
    ($(($($flavor:tt)*) $string:literal)*) => (
        impl LinkerFlavorCli {
            const fn all() -> &'static [LinkerFlavorCli] {
                &[$($($flavor)*,)*]
            }

            pub const fn one_of() -> &'static str {
                concat!("one of: ", $($string, " ",)*)
            }

            pub fn from_str(s: &str) -> Option<LinkerFlavorCli> {
                Some(match s {
                    $($string => $($flavor)*,)*
                    _ => return None,
                })
            }

            pub fn desc(self) -> &'static str {
                match self {
                    $($($flavor)* => $string,)*
                }
            }
        }
    )
}

linker_flavor_cli_impls! {
    (LinkerFlavorCli::Gnu(Cc::No, Lld::No)) "gnu"
    (LinkerFlavorCli::Gnu(Cc::No, Lld::Yes)) "gnu-lld"
    (LinkerFlavorCli::Gnu(Cc::Yes, Lld::No)) "gnu-cc"
    (LinkerFlavorCli::Gnu(Cc::Yes, Lld::Yes)) "gnu-lld-cc"
    (LinkerFlavorCli::Darwin(Cc::No, Lld::No)) "darwin"
    (LinkerFlavorCli::Darwin(Cc::No, Lld::Yes)) "darwin-lld"
    (LinkerFlavorCli::Darwin(Cc::Yes, Lld::No)) "darwin-cc"
    (LinkerFlavorCli::Darwin(Cc::Yes, Lld::Yes)) "darwin-lld-cc"
    (LinkerFlavorCli::WasmLld(Cc::No)) "wasm-lld"
    (LinkerFlavorCli::WasmLld(Cc::Yes)) "wasm-lld-cc"
    (LinkerFlavorCli::Unix(Cc::No)) "unix"
    (LinkerFlavorCli::Unix(Cc::Yes)) "unix-cc"
    (LinkerFlavorCli::Msvc(Lld::Yes)) "msvc-lld"
    (LinkerFlavorCli::Msvc(Lld::No)) "msvc"
    (LinkerFlavorCli::EmCc) "em-cc"
    (LinkerFlavorCli::Bpf) "bpf"
    (LinkerFlavorCli::Ptx) "ptx"

    // Below: legacy stable values
    (LinkerFlavorCli::Gcc) "gcc"
    (LinkerFlavorCli::Ld) "ld"
    (LinkerFlavorCli::Lld(LldFlavor::Ld)) "ld.lld"
    (LinkerFlavorCli::Lld(LldFlavor::Ld64)) "ld64.lld"
    (LinkerFlavorCli::Lld(LldFlavor::Link)) "lld-link"
    (LinkerFlavorCli::Lld(LldFlavor::Wasm)) "wasm-ld"
    (LinkerFlavorCli::Em) "em"
    (LinkerFlavorCli::BpfLinker) "bpf-linker"
    (LinkerFlavorCli::PtxLinker) "ptx-linker"
}

impl ToJson for LinkerFlavorCli {
    fn to_json(&self) -> Json {
        self.desc().to_json()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash, Encodable, Decodable, HashStable_Generic)]
pub enum PanicStrategy {
    Unwind,
    Abort,
}

impl PanicStrategy {
    pub fn desc(&self) -> &str {
        match *self {
            PanicStrategy::Unwind => "unwind",
            PanicStrategy::Abort => "abort",
        }
    }

    pub const fn desc_symbol(&self) -> Symbol {
        match *self {
            PanicStrategy::Unwind => sym::unwind,
            PanicStrategy::Abort => sym::abort,
        }
    }

    pub const fn all() -> [Symbol; 2] {
        [Self::Abort.desc_symbol(), Self::Unwind.desc_symbol()]
    }
}

impl ToJson for PanicStrategy {
    fn to_json(&self) -> Json {
        match *self {
            PanicStrategy::Abort => "abort".to_json(),
            PanicStrategy::Unwind => "unwind".to_json(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum RelroLevel {
    Full,
    Partial,
    Off,
    None,
}

impl RelroLevel {
    pub fn desc(&self) -> &str {
        match *self {
            RelroLevel::Full => "full",
            RelroLevel::Partial => "partial",
            RelroLevel::Off => "off",
            RelroLevel::None => "none",
        }
    }
}

impl FromStr for RelroLevel {
    type Err = ();

    fn from_str(s: &str) -> Result<RelroLevel, ()> {
        match s {
            "full" => Ok(RelroLevel::Full),
            "partial" => Ok(RelroLevel::Partial),
            "off" => Ok(RelroLevel::Off),
            "none" => Ok(RelroLevel::None),
            _ => Err(()),
        }
    }
}

impl ToJson for RelroLevel {
    fn to_json(&self) -> Json {
        match *self {
            RelroLevel::Full => "full".to_json(),
            RelroLevel::Partial => "partial".to_json(),
            RelroLevel::Off => "off".to_json(),
            RelroLevel::None => "None".to_json(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum MergeFunctions {
    Disabled,
    Trampolines,
    Aliases,
}

impl MergeFunctions {
    pub fn desc(&self) -> &str {
        match *self {
            MergeFunctions::Disabled => "disabled",
            MergeFunctions::Trampolines => "trampolines",
            MergeFunctions::Aliases => "aliases",
        }
    }
}

impl FromStr for MergeFunctions {
    type Err = ();

    fn from_str(s: &str) -> Result<MergeFunctions, ()> {
        match s {
            "disabled" => Ok(MergeFunctions::Disabled),
            "trampolines" => Ok(MergeFunctions::Trampolines),
            "aliases" => Ok(MergeFunctions::Aliases),
            _ => Err(()),
        }
    }
}

impl ToJson for MergeFunctions {
    fn to_json(&self) -> Json {
        match *self {
            MergeFunctions::Disabled => "disabled".to_json(),
            MergeFunctions::Trampolines => "trampolines".to_json(),
            MergeFunctions::Aliases => "aliases".to_json(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum RelocModel {
    Static,
    Pic,
    Pie,
    DynamicNoPic,
    Ropi,
    Rwpi,
    RopiRwpi,
}

impl FromStr for RelocModel {
    type Err = ();

    fn from_str(s: &str) -> Result<RelocModel, ()> {
        Ok(match s {
            "static" => RelocModel::Static,
            "pic" => RelocModel::Pic,
            "pie" => RelocModel::Pie,
            "dynamic-no-pic" => RelocModel::DynamicNoPic,
            "ropi" => RelocModel::Ropi,
            "rwpi" => RelocModel::Rwpi,
            "ropi-rwpi" => RelocModel::RopiRwpi,
            _ => return Err(()),
        })
    }
}

impl ToJson for RelocModel {
    fn to_json(&self) -> Json {
        match *self {
            RelocModel::Static => "static",
            RelocModel::Pic => "pic",
            RelocModel::Pie => "pie",
            RelocModel::DynamicNoPic => "dynamic-no-pic",
            RelocModel::Ropi => "ropi",
            RelocModel::Rwpi => "rwpi",
            RelocModel::RopiRwpi => "ropi-rwpi",
        }
        .to_json()
    }
}

#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum CodeModel {
    Tiny,
    Small,
    Kernel,
    Medium,
    Large,
}

impl FromStr for CodeModel {
    type Err = ();

    fn from_str(s: &str) -> Result<CodeModel, ()> {
        Ok(match s {
            "tiny" => CodeModel::Tiny,
            "small" => CodeModel::Small,
            "kernel" => CodeModel::Kernel,
            "medium" => CodeModel::Medium,
            "large" => CodeModel::Large,
            _ => return Err(()),
        })
    }
}

impl ToJson for CodeModel {
    fn to_json(&self) -> Json {
        match *self {
            CodeModel::Tiny => "tiny",
            CodeModel::Small => "small",
            CodeModel::Kernel => "kernel",
            CodeModel::Medium => "medium",
            CodeModel::Large => "large",
        }
        .to_json()
    }
}

#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum TlsModel {
    GeneralDynamic,
    LocalDynamic,
    InitialExec,
    LocalExec,
}

impl FromStr for TlsModel {
    type Err = ();

    fn from_str(s: &str) -> Result<TlsModel, ()> {
        Ok(match s {
            // Note the difference "general" vs "global" difference. The model name is "general",
            // but the user-facing option name is "global" for consistency with other compilers.
            "global-dynamic" => TlsModel::GeneralDynamic,
            "local-dynamic" => TlsModel::LocalDynamic,
            "initial-exec" => TlsModel::InitialExec,
            "local-exec" => TlsModel::LocalExec,
            _ => return Err(()),
        })
    }
}

impl ToJson for TlsModel {
    fn to_json(&self) -> Json {
        match *self {
            TlsModel::GeneralDynamic => "global-dynamic",
            TlsModel::LocalDynamic => "local-dynamic",
            TlsModel::InitialExec => "initial-exec",
            TlsModel::LocalExec => "local-exec",
        }
        .to_json()
    }
}

/// Everything is flattened to a single enum to make the json encoding/decoding less annoying.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LinkOutputKind {
    /// Dynamically linked non position-independent executable.
    DynamicNoPicExe,
    /// Dynamically linked position-independent executable.
    DynamicPicExe,
    /// Statically linked non position-independent executable.
    StaticNoPicExe,
    /// Statically linked position-independent executable.
    StaticPicExe,
    /// Regular dynamic library ("dynamically linked").
    DynamicDylib,
    /// Dynamic library with bundled libc ("statically linked").
    StaticDylib,
    /// WASI module with a lifetime past the _initialize entry point
    WasiReactorExe,
}

impl LinkOutputKind {
    fn as_str(&self) -> &'static str {
        match self {
            LinkOutputKind::DynamicNoPicExe => "dynamic-nopic-exe",
            LinkOutputKind::DynamicPicExe => "dynamic-pic-exe",
            LinkOutputKind::StaticNoPicExe => "static-nopic-exe",
            LinkOutputKind::StaticPicExe => "static-pic-exe",
            LinkOutputKind::DynamicDylib => "dynamic-dylib",
            LinkOutputKind::StaticDylib => "static-dylib",
            LinkOutputKind::WasiReactorExe => "wasi-reactor-exe",
        }
    }

    pub(super) fn from_str(s: &str) -> Option<LinkOutputKind> {
        Some(match s {
            "dynamic-nopic-exe" => LinkOutputKind::DynamicNoPicExe,
            "dynamic-pic-exe" => LinkOutputKind::DynamicPicExe,
            "static-nopic-exe" => LinkOutputKind::StaticNoPicExe,
            "static-pic-exe" => LinkOutputKind::StaticPicExe,
            "dynamic-dylib" => LinkOutputKind::DynamicDylib,
            "static-dylib" => LinkOutputKind::StaticDylib,
            "wasi-reactor-exe" => LinkOutputKind::WasiReactorExe,
            _ => return None,
        })
    }

    pub fn can_link_dylib(self) -> bool {
        match self {
            LinkOutputKind::StaticNoPicExe | LinkOutputKind::StaticPicExe => false,
            LinkOutputKind::DynamicNoPicExe
            | LinkOutputKind::DynamicPicExe
            | LinkOutputKind::DynamicDylib
            | LinkOutputKind::StaticDylib
            | LinkOutputKind::WasiReactorExe => true,
        }
    }
}

impl fmt::Display for LinkOutputKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

pub type LinkArgs = BTreeMap<LinkerFlavor, Vec<StaticCow<str>>>;
pub type LinkArgsCli = BTreeMap<LinkerFlavorCli, Vec<StaticCow<str>>>;

/// Which kind of debuginfo does the target use?
///
/// Useful in determining whether a target supports Split DWARF (a target with
/// `DebuginfoKind::Dwarf` and supporting `SplitDebuginfo::Unpacked` for example).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum DebuginfoKind {
    /// DWARF debuginfo (such as that used on `x86_64_unknown_linux_gnu`).
    #[default]
    Dwarf,
    /// DWARF debuginfo in dSYM files (such as on Apple platforms).
    DwarfDsym,
    /// Program database files (such as on Windows).
    Pdb,
}

impl DebuginfoKind {
    fn as_str(&self) -> &'static str {
        match self {
            DebuginfoKind::Dwarf => "dwarf",
            DebuginfoKind::DwarfDsym => "dwarf-dsym",
            DebuginfoKind::Pdb => "pdb",
        }
    }
}

impl FromStr for DebuginfoKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        Ok(match s {
            "dwarf" => DebuginfoKind::Dwarf,
            "dwarf-dsym" => DebuginfoKind::DwarfDsym,
            "pdb" => DebuginfoKind::Pdb,
            _ => return Err(()),
        })
    }
}

impl ToJson for DebuginfoKind {
    fn to_json(&self) -> Json {
        self.as_str().to_json()
    }
}

impl fmt::Display for DebuginfoKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum SplitDebuginfo {
    /// Split debug-information is disabled, meaning that on supported platforms
    /// you can find all debug information in the executable itself. This is
    /// only supported for ELF effectively.
    ///
    /// * Windows - not supported
    /// * macOS - don't run `dsymutil`
    /// * ELF - `.debug_*` sections
    #[default]
    Off,

    /// Split debug-information can be found in a "packed" location separate
    /// from the final artifact. This is supported on all platforms.
    ///
    /// * Windows - `*.pdb`
    /// * macOS - `*.dSYM` (run `dsymutil`)
    /// * ELF - `*.dwp` (run `thorin`)
    Packed,

    /// Split debug-information can be found in individual object files on the
    /// filesystem. The main executable may point to the object files.
    ///
    /// * Windows - not supported
    /// * macOS - supported, scattered object files
    /// * ELF - supported, scattered `*.dwo` or `*.o` files (see `SplitDwarfKind`)
    Unpacked,
}

impl SplitDebuginfo {
    fn as_str(&self) -> &'static str {
        match self {
            SplitDebuginfo::Off => "off",
            SplitDebuginfo::Packed => "packed",
            SplitDebuginfo::Unpacked => "unpacked",
        }
    }
}

impl FromStr for SplitDebuginfo {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        Ok(match s {
            "off" => SplitDebuginfo::Off,
            "unpacked" => SplitDebuginfo::Unpacked,
            "packed" => SplitDebuginfo::Packed,
            _ => return Err(()),
        })
    }
}

impl ToJson for SplitDebuginfo {
    fn to_json(&self) -> Json {
        self.as_str().to_json()
    }
}

impl fmt::Display for SplitDebuginfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StackProbeType {
    /// Don't emit any stack probes.
    None,
    /// It is harmless to use this option even on targets that do not have backend support for
    /// stack probes as the failure mode is the same as if no stack-probe option was specified in
    /// the first place.
    Inline,
    /// Call `__rust_probestack` whenever stack needs to be probed.
    Call,
    /// Use inline option for LLVM versions later than specified in `min_llvm_version_for_inline`
    /// and call `__rust_probestack` otherwise.
    InlineOrCall { min_llvm_version_for_inline: (u32, u32, u32) },
}

impl StackProbeType {
    // LLVM X86 targets (ix86 and x86_64) can use inline-asm stack probes starting with LLVM 16.
    // Notable past issues were rust#83139 (fixed in 14) and rust#84667 (fixed in 16).
    const X86: Self = Self::InlineOrCall { min_llvm_version_for_inline: (16, 0, 0) };

    fn from_json(json: &Json) -> Result<Self, String> {
        let object = json.as_object().ok_or_else(|| "expected a JSON object")?;
        let kind = object
            .get("kind")
            .and_then(|o| o.as_str())
            .ok_or_else(|| "expected `kind` to be a string")?;
        match kind {
            "none" => Ok(StackProbeType::None),
            "inline" => Ok(StackProbeType::Inline),
            "call" => Ok(StackProbeType::Call),
            "inline-or-call" => {
                let min_version = object
                    .get("min-llvm-version-for-inline")
                    .and_then(|o| o.as_array())
                    .ok_or_else(|| "expected `min-llvm-version-for-inline` to be an array")?;
                let mut iter = min_version.into_iter().map(|v| {
                    let int = v.as_u64().ok_or_else(
                        || "expected `min-llvm-version-for-inline` values to be integers",
                    )?;
                    u32::try_from(int)
                        .map_err(|_| "`min-llvm-version-for-inline` values don't convert to u32")
                });
                let min_llvm_version_for_inline = (
                    iter.next().unwrap_or(Ok(11))?,
                    iter.next().unwrap_or(Ok(0))?,
                    iter.next().unwrap_or(Ok(0))?,
                );
                Ok(StackProbeType::InlineOrCall { min_llvm_version_for_inline })
            }
            _ => Err(String::from(
                "`kind` expected to be one of `none`, `inline`, `call` or `inline-or-call`",
            )),
        }
    }
}

impl ToJson for StackProbeType {
    fn to_json(&self) -> Json {
        Json::Object(match self {
            StackProbeType::None => {
                [(String::from("kind"), "none".to_json())].into_iter().collect()
            }
            StackProbeType::Inline => {
                [(String::from("kind"), "inline".to_json())].into_iter().collect()
            }
            StackProbeType::Call => {
                [(String::from("kind"), "call".to_json())].into_iter().collect()
            }
            StackProbeType::InlineOrCall { min_llvm_version_for_inline: (maj, min, patch) } => [
                (String::from("kind"), "inline-or-call".to_json()),
                (
                    String::from("min-llvm-version-for-inline"),
                    Json::Array(vec![maj.to_json(), min.to_json(), patch.to_json()]),
                ),
            ]
            .into_iter()
            .collect(),
        })
    }
}

bitflags::bitflags! {
    #[derive(Default, Encodable, Decodable)]
    pub struct SanitizerSet: u16 {
        const ADDRESS = 1 << 0;
        const LEAK    = 1 << 1;
        const MEMORY  = 1 << 2;
        const THREAD  = 1 << 3;
        const HWADDRESS = 1 << 4;
        const CFI     = 1 << 5;
        const MEMTAG  = 1 << 6;
        const SHADOWCALLSTACK = 1 << 7;
        const KCFI    = 1 << 8;
        const KERNELADDRESS = 1 << 9;
        const SAFESTACK = 1 << 10;
    }
}

impl SanitizerSet {
    /// Return sanitizer's name
    ///
    /// Returns none if the flags is a set of sanitizers numbering not exactly one.
    pub fn as_str(self) -> Option<&'static str> {
        Some(match self {
            SanitizerSet::ADDRESS => "address",
            SanitizerSet::CFI => "cfi",
            SanitizerSet::KCFI => "kcfi",
            SanitizerSet::KERNELADDRESS => "kernel-address",
            SanitizerSet::LEAK => "leak",
            SanitizerSet::MEMORY => "memory",
            SanitizerSet::MEMTAG => "memtag",
            SanitizerSet::SAFESTACK => "safestack",
            SanitizerSet::SHADOWCALLSTACK => "shadow-call-stack",
            SanitizerSet::THREAD => "thread",
            SanitizerSet::HWADDRESS => "hwaddress",
            _ => return None,
        })
    }
}

/// Formats a sanitizer set as a comma separated list of sanitizers' names.
impl fmt::Display for SanitizerSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for s in *self {
            let name = s.as_str().unwrap_or_else(|| panic!("unrecognized sanitizer {s:?}"));
            if !first {
                f.write_str(", ")?;
            }
            f.write_str(name)?;
            first = false;
        }
        Ok(())
    }
}

impl IntoIterator for SanitizerSet {
    type Item = SanitizerSet;
    type IntoIter = std::vec::IntoIter<SanitizerSet>;

    fn into_iter(self) -> Self::IntoIter {
        [
            SanitizerSet::ADDRESS,
            SanitizerSet::CFI,
            SanitizerSet::KCFI,
            SanitizerSet::LEAK,
            SanitizerSet::MEMORY,
            SanitizerSet::MEMTAG,
            SanitizerSet::SHADOWCALLSTACK,
            SanitizerSet::THREAD,
            SanitizerSet::HWADDRESS,
            SanitizerSet::KERNELADDRESS,
            SanitizerSet::SAFESTACK,
        ]
        .iter()
        .copied()
        .filter(|&s| self.contains(s))
        .collect::<Vec<_>>()
        .into_iter()
    }
}

impl<CTX> HashStable<CTX> for SanitizerSet {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        self.bits().hash_stable(ctx, hasher);
    }
}

impl ToJson for SanitizerSet {
    fn to_json(&self) -> Json {
        self.into_iter()
            .map(|v| Some(v.as_str()?.to_json()))
            .collect::<Option<Vec<_>>>()
            .unwrap_or_default()
            .to_json()
    }
}

#[derive(Clone, Copy, PartialEq, Hash, Debug)]
pub enum FramePointer {
    /// Forces the machine code generator to always preserve the frame pointers.
    Always,
    /// Forces the machine code generator to preserve the frame pointers except for the leaf
    /// functions (i.e. those that don't call other functions).
    NonLeaf,
    /// Allows the machine code generator to omit the frame pointers.
    ///
    /// This option does not guarantee that the frame pointers will be omitted.
    MayOmit,
}

impl FromStr for FramePointer {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        Ok(match s {
            "always" => Self::Always,
            "non-leaf" => Self::NonLeaf,
            "may-omit" => Self::MayOmit,
            _ => return Err(()),
        })
    }
}

impl ToJson for FramePointer {
    fn to_json(&self) -> Json {
        match *self {
            Self::Always => "always",
            Self::NonLeaf => "non-leaf",
            Self::MayOmit => "may-omit",
        }
        .to_json()
    }
}

/// Controls use of stack canaries.
#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq)]
pub enum StackProtector {
    /// Disable stack canary generation.
    None,

    /// On LLVM, mark all generated LLVM functions with the `ssp` attribute (see
    /// llvm/docs/LangRef.rst). This triggers stack canary generation in
    /// functions which contain an array of a byte-sized type with more than
    /// eight elements.
    Basic,

    /// On LLVM, mark all generated LLVM functions with the `sspstrong`
    /// attribute (see llvm/docs/LangRef.rst). This triggers stack canary
    /// generation in functions which either contain an array, or which take
    /// the address of a local variable.
    Strong,

    /// Generate stack canaries in all functions.
    All,
}

impl StackProtector {
    fn as_str(&self) -> &'static str {
        match self {
            StackProtector::None => "none",
            StackProtector::Basic => "basic",
            StackProtector::Strong => "strong",
            StackProtector::All => "all",
        }
    }
}

impl FromStr for StackProtector {
    type Err = ();

    fn from_str(s: &str) -> Result<StackProtector, ()> {
        Ok(match s {
            "none" => StackProtector::None,
            "basic" => StackProtector::Basic,
            "strong" => StackProtector::Strong,
            "all" => StackProtector::All,
            _ => return Err(()),
        })
    }
}

impl fmt::Display for StackProtector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

macro_rules! supported_targets {
    ( $(($triple:literal, $module:ident),)+ ) => {
        $(mod $module;)+

        /// List of supported targets
        pub const TARGETS: &[&str] = &[$($triple),+];

        fn load_builtin(target: &str) -> Option<Target> {
            let mut t = match target {
                $( $triple => $module::target(), )+
                _ => return None,
            };
            t.is_builtin = true;
            debug!("got builtin target: {:?}", t);
            Some(t)
        }

        #[cfg(test)]
        mod tests {
            mod tests_impl;

            // Cannot put this into a separate file without duplication, make an exception.
            $(
                #[test] // `#[test]`
                fn $module() {
                    tests_impl::test_target(super::$module::target());
                }
            )+
        }
    };
}

supported_targets! {
    ("x86_64-unknown-linux-gnu", x86_64_unknown_linux_gnu),
    ("x86_64-unknown-linux-gnux32", x86_64_unknown_linux_gnux32),
    ("i686-unknown-linux-gnu", i686_unknown_linux_gnu),
    ("i586-unknown-linux-gnu", i586_unknown_linux_gnu),
    ("loongarch64-unknown-linux-gnu", loongarch64_unknown_linux_gnu),
    ("m68k-unknown-linux-gnu", m68k_unknown_linux_gnu),
    ("mips-unknown-linux-gnu", mips_unknown_linux_gnu),
    ("mips64-unknown-linux-gnuabi64", mips64_unknown_linux_gnuabi64),
    ("mips64el-unknown-linux-gnuabi64", mips64el_unknown_linux_gnuabi64),
    ("mipsisa32r6-unknown-linux-gnu", mipsisa32r6_unknown_linux_gnu),
    ("mipsisa32r6el-unknown-linux-gnu", mipsisa32r6el_unknown_linux_gnu),
    ("mipsisa64r6-unknown-linux-gnuabi64", mipsisa64r6_unknown_linux_gnuabi64),
    ("mipsisa64r6el-unknown-linux-gnuabi64", mipsisa64r6el_unknown_linux_gnuabi64),
    ("mipsel-unknown-linux-gnu", mipsel_unknown_linux_gnu),
    ("powerpc-unknown-linux-gnu", powerpc_unknown_linux_gnu),
    ("powerpc-unknown-linux-gnuspe", powerpc_unknown_linux_gnuspe),
    ("powerpc-unknown-linux-musl", powerpc_unknown_linux_musl),
    ("powerpc64-ibm-aix", powerpc64_ibm_aix),
    ("powerpc64-unknown-linux-gnu", powerpc64_unknown_linux_gnu),
    ("powerpc64-unknown-linux-musl", powerpc64_unknown_linux_musl),
    ("powerpc64le-unknown-linux-gnu", powerpc64le_unknown_linux_gnu),
    ("powerpc64le-unknown-linux-musl", powerpc64le_unknown_linux_musl),
    ("s390x-unknown-linux-gnu", s390x_unknown_linux_gnu),
    ("s390x-unknown-linux-musl", s390x_unknown_linux_musl),
    ("sparc-unknown-linux-gnu", sparc_unknown_linux_gnu),
    ("sparc64-unknown-linux-gnu", sparc64_unknown_linux_gnu),
    ("arm-unknown-linux-gnueabi", arm_unknown_linux_gnueabi),
    ("arm-unknown-linux-gnueabihf", arm_unknown_linux_gnueabihf),
    ("armeb-unknown-linux-gnueabi", armeb_unknown_linux_gnueabi),
    ("arm-unknown-linux-musleabi", arm_unknown_linux_musleabi),
    ("arm-unknown-linux-musleabihf", arm_unknown_linux_musleabihf),
    ("armv4t-unknown-linux-gnueabi", armv4t_unknown_linux_gnueabi),
    ("armv5te-unknown-linux-gnueabi", armv5te_unknown_linux_gnueabi),
    ("armv5te-unknown-linux-musleabi", armv5te_unknown_linux_musleabi),
    ("armv5te-unknown-linux-uclibceabi", armv5te_unknown_linux_uclibceabi),
    ("armv7-unknown-linux-gnueabi", armv7_unknown_linux_gnueabi),
    ("armv7-unknown-linux-gnueabihf", armv7_unknown_linux_gnueabihf),
    ("thumbv7neon-unknown-linux-gnueabihf", thumbv7neon_unknown_linux_gnueabihf),
    ("thumbv7neon-unknown-linux-musleabihf", thumbv7neon_unknown_linux_musleabihf),
    ("armv7-unknown-linux-musleabi", armv7_unknown_linux_musleabi),
    ("armv7-unknown-linux-musleabihf", armv7_unknown_linux_musleabihf),
    ("aarch64-unknown-linux-gnu", aarch64_unknown_linux_gnu),
    ("aarch64-unknown-linux-musl", aarch64_unknown_linux_musl),
    ("x86_64-unknown-linux-musl", x86_64_unknown_linux_musl),
    ("i686-unknown-linux-musl", i686_unknown_linux_musl),
    ("i586-unknown-linux-musl", i586_unknown_linux_musl),
    ("mips-unknown-linux-musl", mips_unknown_linux_musl),
    ("mipsel-unknown-linux-musl", mipsel_unknown_linux_musl),
    ("mips64-unknown-linux-muslabi64", mips64_unknown_linux_muslabi64),
    ("mips64el-unknown-linux-muslabi64", mips64el_unknown_linux_muslabi64),
    ("hexagon-unknown-linux-musl", hexagon_unknown_linux_musl),

    ("mips-unknown-linux-uclibc", mips_unknown_linux_uclibc),
    ("mipsel-unknown-linux-uclibc", mipsel_unknown_linux_uclibc),

    ("i686-linux-android", i686_linux_android),
    ("x86_64-linux-android", x86_64_linux_android),
    ("arm-linux-androideabi", arm_linux_androideabi),
    ("armv7-linux-androideabi", armv7_linux_androideabi),
    ("thumbv7neon-linux-androideabi", thumbv7neon_linux_androideabi),
    ("aarch64-linux-android", aarch64_linux_android),

    ("aarch64-unknown-freebsd", aarch64_unknown_freebsd),
    ("armv6-unknown-freebsd", armv6_unknown_freebsd),
    ("armv7-unknown-freebsd", armv7_unknown_freebsd),
    ("i686-unknown-freebsd", i686_unknown_freebsd),
    ("powerpc-unknown-freebsd", powerpc_unknown_freebsd),
    ("powerpc64-unknown-freebsd", powerpc64_unknown_freebsd),
    ("powerpc64le-unknown-freebsd", powerpc64le_unknown_freebsd),
    ("riscv64gc-unknown-freebsd", riscv64gc_unknown_freebsd),
    ("x86_64-unknown-freebsd", x86_64_unknown_freebsd),

    ("x86_64-unknown-dragonfly", x86_64_unknown_dragonfly),

    ("aarch64-unknown-openbsd", aarch64_unknown_openbsd),
    ("i686-unknown-openbsd", i686_unknown_openbsd),
    ("powerpc-unknown-openbsd", powerpc_unknown_openbsd),
    ("powerpc64-unknown-openbsd", powerpc64_unknown_openbsd),
    ("riscv64gc-unknown-openbsd", riscv64gc_unknown_openbsd),
    ("sparc64-unknown-openbsd", sparc64_unknown_openbsd),
    ("x86_64-unknown-openbsd", x86_64_unknown_openbsd),

    ("aarch64-unknown-netbsd", aarch64_unknown_netbsd),
    ("aarch64_be-unknown-netbsd", aarch64_be_unknown_netbsd),
    ("armv6-unknown-netbsd-eabihf", armv6_unknown_netbsd_eabihf),
    ("armv7-unknown-netbsd-eabihf", armv7_unknown_netbsd_eabihf),
    ("i686-unknown-netbsd", i686_unknown_netbsd),
    ("powerpc-unknown-netbsd", powerpc_unknown_netbsd),
    ("riscv64gc-unknown-netbsd", riscv64gc_unknown_netbsd),
    ("sparc64-unknown-netbsd", sparc64_unknown_netbsd),
    ("x86_64-unknown-netbsd", x86_64_unknown_netbsd),

    ("i686-unknown-haiku", i686_unknown_haiku),
    ("x86_64-unknown-haiku", x86_64_unknown_haiku),

    ("aarch64-apple-darwin", aarch64_apple_darwin),
    ("x86_64-apple-darwin", x86_64_apple_darwin),
    ("x86_64h-apple-darwin", x86_64h_apple_darwin),
    ("i686-apple-darwin", i686_apple_darwin),

    // FIXME(#106649): Remove aarch64-fuchsia in favor of aarch64-unknown-fuchsia
    ("aarch64-fuchsia", aarch64_fuchsia),
    ("aarch64-unknown-fuchsia", aarch64_unknown_fuchsia),
    ("riscv64gc-unknown-fuchsia", riscv64gc_unknown_fuchsia),
    // FIXME(#106649): Remove x86_64-fuchsia in favor of x86_64-unknown-fuchsia
    ("x86_64-fuchsia", x86_64_fuchsia),
    ("x86_64-unknown-fuchsia", x86_64_unknown_fuchsia),

    ("avr-unknown-gnu-atmega328", avr_unknown_gnu_atmega328),

    ("x86_64-unknown-l4re-uclibc", x86_64_unknown_l4re_uclibc),

    ("aarch64-unknown-redox", aarch64_unknown_redox),
    ("x86_64-unknown-redox", x86_64_unknown_redox),

    ("i386-apple-ios", i386_apple_ios),
    ("x86_64-apple-ios", x86_64_apple_ios),
    ("aarch64-apple-ios", aarch64_apple_ios),
    ("armv7-apple-ios", armv7_apple_ios),
    ("armv7s-apple-ios", armv7s_apple_ios),
    ("x86_64-apple-ios-macabi", x86_64_apple_ios_macabi),
    ("aarch64-apple-ios-macabi", aarch64_apple_ios_macabi),
    ("aarch64-apple-ios-sim", aarch64_apple_ios_sim),
    ("aarch64-apple-tvos", aarch64_apple_tvos),
    ("x86_64-apple-tvos", x86_64_apple_tvos),

    ("armv7k-apple-watchos", armv7k_apple_watchos),
    ("arm64_32-apple-watchos", arm64_32_apple_watchos),
    ("x86_64-apple-watchos-sim", x86_64_apple_watchos_sim),
    ("aarch64-apple-watchos-sim", aarch64_apple_watchos_sim),

    ("armebv7r-none-eabi", armebv7r_none_eabi),
    ("armebv7r-none-eabihf", armebv7r_none_eabihf),
    ("armv7r-none-eabi", armv7r_none_eabi),
    ("armv7r-none-eabihf", armv7r_none_eabihf),

    ("x86_64-pc-solaris", x86_64_pc_solaris),
    ("x86_64-sun-solaris", x86_64_sun_solaris),
    ("sparcv9-sun-solaris", sparcv9_sun_solaris),

    ("x86_64-unknown-illumos", x86_64_unknown_illumos),

    ("x86_64-pc-windows-gnu", x86_64_pc_windows_gnu),
    ("i686-pc-windows-gnu", i686_pc_windows_gnu),
    ("i686-uwp-windows-gnu", i686_uwp_windows_gnu),
    ("x86_64-uwp-windows-gnu", x86_64_uwp_windows_gnu),

    ("aarch64-pc-windows-gnullvm", aarch64_pc_windows_gnullvm),
    ("x86_64-pc-windows-gnullvm", x86_64_pc_windows_gnullvm),

    ("aarch64-pc-windows-msvc", aarch64_pc_windows_msvc),
    ("aarch64-uwp-windows-msvc", aarch64_uwp_windows_msvc),
    ("x86_64-pc-windows-msvc", x86_64_pc_windows_msvc),
    ("x86_64-uwp-windows-msvc", x86_64_uwp_windows_msvc),
    ("i686-pc-windows-msvc", i686_pc_windows_msvc),
    ("i686-uwp-windows-msvc", i686_uwp_windows_msvc),
    ("i586-pc-windows-msvc", i586_pc_windows_msvc),
    ("thumbv7a-pc-windows-msvc", thumbv7a_pc_windows_msvc),
    ("thumbv7a-uwp-windows-msvc", thumbv7a_uwp_windows_msvc),

    ("asmjs-unknown-emscripten", asmjs_unknown_emscripten),
    ("wasm32-unknown-emscripten", wasm32_unknown_emscripten),
    ("wasm32-unknown-unknown", wasm32_unknown_unknown),
    ("wasm32-wasi", wasm32_wasi),
    ("wasm64-unknown-unknown", wasm64_unknown_unknown),

    ("thumbv6m-none-eabi", thumbv6m_none_eabi),
    ("thumbv7m-none-eabi", thumbv7m_none_eabi),
    ("thumbv7em-none-eabi", thumbv7em_none_eabi),
    ("thumbv7em-none-eabihf", thumbv7em_none_eabihf),
    ("thumbv8m.base-none-eabi", thumbv8m_base_none_eabi),
    ("thumbv8m.main-none-eabi", thumbv8m_main_none_eabi),
    ("thumbv8m.main-none-eabihf", thumbv8m_main_none_eabihf),

    ("armv7a-none-eabi", armv7a_none_eabi),
    ("armv7a-none-eabihf", armv7a_none_eabihf),

    ("msp430-none-elf", msp430_none_elf),

    ("aarch64-unknown-hermit", aarch64_unknown_hermit),
    ("x86_64-unknown-hermit", x86_64_unknown_hermit),

    ("riscv32i-unknown-none-elf", riscv32i_unknown_none_elf),
    ("riscv32im-unknown-none-elf", riscv32im_unknown_none_elf),
    ("riscv32imc-unknown-none-elf", riscv32imc_unknown_none_elf),
    ("riscv32imc-esp-espidf", riscv32imc_esp_espidf),
    ("riscv32imac-esp-espidf", riscv32imac_esp_espidf),
    ("riscv32imac-unknown-none-elf", riscv32imac_unknown_none_elf),
    ("riscv32imac-unknown-xous-elf", riscv32imac_unknown_xous_elf),
    ("riscv32gc-unknown-linux-gnu", riscv32gc_unknown_linux_gnu),
    ("riscv32gc-unknown-linux-musl", riscv32gc_unknown_linux_musl),
    ("riscv64imac-unknown-none-elf", riscv64imac_unknown_none_elf),
    ("riscv64gc-unknown-none-elf", riscv64gc_unknown_none_elf),
    ("riscv64gc-unknown-linux-gnu", riscv64gc_unknown_linux_gnu),
    ("riscv64gc-unknown-linux-musl", riscv64gc_unknown_linux_musl),

    ("loongarch64-unknown-none", loongarch64_unknown_none),
    ("loongarch64-unknown-none-softfloat", loongarch64_unknown_none_softfloat),

    ("aarch64-unknown-none", aarch64_unknown_none),
    ("aarch64-unknown-none-softfloat", aarch64_unknown_none_softfloat),

    ("x86_64-fortanix-unknown-sgx", x86_64_fortanix_unknown_sgx),

    ("x86_64-unknown-uefi", x86_64_unknown_uefi),
    ("i686-unknown-uefi", i686_unknown_uefi),
    ("aarch64-unknown-uefi", aarch64_unknown_uefi),

    ("nvptx64-nvidia-cuda", nvptx64_nvidia_cuda),

    ("i686-wrs-vxworks", i686_wrs_vxworks),
    ("x86_64-wrs-vxworks", x86_64_wrs_vxworks),
    ("armv7-wrs-vxworks-eabihf", armv7_wrs_vxworks_eabihf),
    ("aarch64-wrs-vxworks", aarch64_wrs_vxworks),
    ("powerpc-wrs-vxworks", powerpc_wrs_vxworks),
    ("powerpc-wrs-vxworks-spe", powerpc_wrs_vxworks_spe),
    ("powerpc64-wrs-vxworks", powerpc64_wrs_vxworks),

    ("aarch64-kmc-solid_asp3", aarch64_kmc_solid_asp3),
    ("armv7a-kmc-solid_asp3-eabi", armv7a_kmc_solid_asp3_eabi),
    ("armv7a-kmc-solid_asp3-eabihf", armv7a_kmc_solid_asp3_eabihf),

    ("mipsel-sony-psp", mipsel_sony_psp),
    ("mipsel-sony-psx", mipsel_sony_psx),
    ("mipsel-unknown-none", mipsel_unknown_none),
    ("thumbv4t-none-eabi", thumbv4t_none_eabi),
    ("armv4t-none-eabi", armv4t_none_eabi),
    ("thumbv5te-none-eabi", thumbv5te_none_eabi),
    ("armv5te-none-eabi", armv5te_none_eabi),

    ("aarch64_be-unknown-linux-gnu", aarch64_be_unknown_linux_gnu),
    ("aarch64-unknown-linux-gnu_ilp32", aarch64_unknown_linux_gnu_ilp32),
    ("aarch64_be-unknown-linux-gnu_ilp32", aarch64_be_unknown_linux_gnu_ilp32),

    ("bpfeb-unknown-none", bpfeb_unknown_none),
    ("bpfel-unknown-none", bpfel_unknown_none),

    ("armv6k-nintendo-3ds", armv6k_nintendo_3ds),

    ("aarch64-nintendo-switch-freestanding", aarch64_nintendo_switch_freestanding),

    ("armv7-sony-vita-newlibeabihf", armv7_sony_vita_newlibeabihf),

    ("armv7-unknown-linux-uclibceabi", armv7_unknown_linux_uclibceabi),
    ("armv7-unknown-linux-uclibceabihf", armv7_unknown_linux_uclibceabihf),

    ("x86_64-unknown-none", x86_64_unknown_none),

    ("mips64-openwrt-linux-musl", mips64_openwrt_linux_musl),

    ("aarch64-unknown-nto-qnx710", aarch64_unknown_nto_qnx_710),
    ("x86_64-pc-nto-qnx710", x86_64_pc_nto_qnx710),
    ("i586-pc-nto-qnx700", i586_pc_nto_qnx700),

    ("aarch64-unknown-linux-ohos", aarch64_unknown_linux_ohos),
    ("armv7-unknown-linux-ohos", armv7_unknown_linux_ohos),
}

/// Cow-Vec-Str: Cow<'static, [Cow<'static, str>]>
macro_rules! cvs {
    () => {
        ::std::borrow::Cow::Borrowed(&[])
    };
    ($($x:expr),+ $(,)?) => {
        ::std::borrow::Cow::Borrowed(&[
            $(
                ::std::borrow::Cow::Borrowed($x),
            )*
        ])
    };
}

pub(crate) use cvs;

/// Warnings encountered when parsing the target `json`.
///
/// Includes fields that weren't recognized and fields that don't have the expected type.
#[derive(Debug, PartialEq)]
pub struct TargetWarnings {
    unused_fields: Vec<String>,
    incorrect_type: Vec<String>,
}

impl TargetWarnings {
    pub fn empty() -> Self {
        Self { unused_fields: Vec::new(), incorrect_type: Vec::new() }
    }

    pub fn warning_messages(&self) -> Vec<String> {
        let mut warnings = vec![];
        if !self.unused_fields.is_empty() {
            warnings.push(format!(
                "target json file contains unused fields: {}",
                self.unused_fields.join(", ")
            ));
        }
        if !self.incorrect_type.is_empty() {
            warnings.push(format!(
                "target json file contains fields whose value doesn't have the correct json type: {}",
                self.incorrect_type.join(", ")
            ));
        }
        warnings
    }
}

/// Everything `rustc` knows about how to compile for a specific target.
///
/// Every field here must be specified, and has no default value.
#[derive(PartialEq, Clone, Debug)]
pub struct Target {
    /// Target triple to pass to LLVM.
    pub llvm_target: StaticCow<str>,
    /// Number of bits in a pointer. Influences the `target_pointer_width` `cfg` variable.
    pub pointer_width: u32,
    /// Architecture to use for ABI considerations. Valid options include: "x86",
    /// "x86_64", "arm", "aarch64", "mips", "powerpc", "powerpc64", and others.
    pub arch: StaticCow<str>,
    /// [Data layout](https://llvm.org/docs/LangRef.html#data-layout) to pass to LLVM.
    pub data_layout: StaticCow<str>,
    /// Optional settings with defaults.
    pub options: TargetOptions,
}

impl Target {
    pub fn parse_data_layout(&self) -> Result<TargetDataLayout, TargetDataLayoutErrors<'_>> {
        let mut dl = TargetDataLayout::parse_from_llvm_datalayout_string(&self.data_layout)?;

        // Perform consistency checks against the Target information.
        if dl.endian != self.endian {
            return Err(TargetDataLayoutErrors::InconsistentTargetArchitecture {
                dl: dl.endian.as_str(),
                target: self.endian.as_str(),
            });
        }

        let target_pointer_width: u64 = self.pointer_width.into();
        if dl.pointer_size.bits() != target_pointer_width {
            return Err(TargetDataLayoutErrors::InconsistentTargetPointerWidth {
                pointer_size: dl.pointer_size.bits(),
                target: self.pointer_width,
            });
        }

        dl.c_enum_min_size = self
            .c_enum_min_bits
            .map_or_else(
                || {
                    self.c_int_width
                        .parse()
                        .map_err(|_| String::from("failed to parse c_int_width"))
                },
                Ok,
            )
            .and_then(|i| Integer::from_size(Size::from_bits(i)))
            .map_err(|err| TargetDataLayoutErrors::InvalidBitsSize { err })?;

        Ok(dl)
    }
}

pub trait HasTargetSpec {
    fn target_spec(&self) -> &Target;
}

impl HasTargetSpec for Target {
    #[inline]
    fn target_spec(&self) -> &Target {
        self
    }
}

type StaticCow<T> = Cow<'static, T>;

/// Optional aspects of a target specification.
///
/// This has an implementation of `Default`, see each field for what the default is. In general,
/// these try to take "minimal defaults" that don't assume anything about the runtime they run in.
///
/// `TargetOptions` as a separate structure is mostly an implementation detail of `Target`
/// construction, all its fields logically belong to `Target` and available from `Target`
/// through `Deref` impls.
#[derive(PartialEq, Clone, Debug)]
pub struct TargetOptions {
    /// Whether the target is built-in or loaded from a custom target specification.
    pub is_builtin: bool,

    /// Used as the `target_endian` `cfg` variable. Defaults to little endian.
    pub endian: Endian,
    /// Width of c_int type. Defaults to "32".
    pub c_int_width: StaticCow<str>,
    /// OS name to use for conditional compilation (`target_os`). Defaults to "none".
    /// "none" implies a bare metal target without `std` library.
    /// A couple of targets having `std` also use "unknown" as an `os` value,
    /// but they are exceptions.
    pub os: StaticCow<str>,
    /// Environment name to use for conditional compilation (`target_env`). Defaults to "".
    pub env: StaticCow<str>,
    /// ABI name to distinguish multiple ABIs on the same OS and architecture. For instance, `"eabi"`
    /// or `"eabihf"`. Defaults to "".
    pub abi: StaticCow<str>,
    /// Vendor name to use for conditional compilation (`target_vendor`). Defaults to "unknown".
    pub vendor: StaticCow<str>,

    /// Linker to invoke
    pub linker: Option<StaticCow<str>>,
    /// Default linker flavor used if `-C linker-flavor` or `-C linker` are not passed
    /// on the command line. Defaults to `LinkerFlavor::Gnu(Cc::Yes, Lld::No)`.
    pub linker_flavor: LinkerFlavor,
    linker_flavor_json: LinkerFlavorCli,
    lld_flavor_json: LldFlavor,
    linker_is_gnu_json: bool,

    /// Objects to link before and after all other object code.
    pub pre_link_objects: CrtObjects,
    pub post_link_objects: CrtObjects,
    /// Same as `(pre|post)_link_objects`, but when self-contained linking mode is enabled.
    pub pre_link_objects_self_contained: CrtObjects,
    pub post_link_objects_self_contained: CrtObjects,
    pub link_self_contained: LinkSelfContainedDefault,

    /// Linker arguments that are passed *before* any user-defined libraries.
    pub pre_link_args: LinkArgs,
    pre_link_args_json: LinkArgsCli,
    /// Linker arguments that are unconditionally passed after any
    /// user-defined but before post-link objects. Standard platform
    /// libraries that should be always be linked to, usually go here.
    pub late_link_args: LinkArgs,
    late_link_args_json: LinkArgsCli,
    /// Linker arguments used in addition to `late_link_args` if at least one
    /// Rust dependency is dynamically linked.
    pub late_link_args_dynamic: LinkArgs,
    late_link_args_dynamic_json: LinkArgsCli,
    /// Linker arguments used in addition to `late_link_args` if all Rust
    /// dependencies are statically linked.
    pub late_link_args_static: LinkArgs,
    late_link_args_static_json: LinkArgsCli,
    /// Linker arguments that are unconditionally passed *after* any
    /// user-defined libraries.
    pub post_link_args: LinkArgs,
    post_link_args_json: LinkArgsCli,

    /// Optional link script applied to `dylib` and `executable` crate types.
    /// This is a string containing the script, not a path. Can only be applied
    /// to linkers where linker flavor matches `LinkerFlavor::Gnu(..)`.
    pub link_script: Option<StaticCow<str>>,
    /// Environment variables to be set for the linker invocation.
    pub link_env: StaticCow<[(StaticCow<str>, StaticCow<str>)]>,
    /// Environment variables to be removed for the linker invocation.
    pub link_env_remove: StaticCow<[StaticCow<str>]>,

    /// Extra arguments to pass to the external assembler (when used)
    pub asm_args: StaticCow<[StaticCow<str>]>,

    /// Default CPU to pass to LLVM. Corresponds to `llc -mcpu=$cpu`. Defaults
    /// to "generic".
    pub cpu: StaticCow<str>,
    /// Default target features to pass to LLVM. These features will *always* be
    /// passed, and cannot be disabled even via `-C`. Corresponds to `llc
    /// -mattr=$features`.
    pub features: StaticCow<str>,
    /// Whether dynamic linking is available on this target. Defaults to false.
    pub dynamic_linking: bool,
    /// Whether dynamic linking can export TLS globals. Defaults to true.
    pub dll_tls_export: bool,
    /// If dynamic linking is available, whether only cdylibs are supported.
    pub only_cdylib: bool,
    /// Whether executables are available on this target. Defaults to true.
    pub executables: bool,
    /// Relocation model to use in object file. Corresponds to `llc
    /// -relocation-model=$relocation_model`. Defaults to `Pic`.
    pub relocation_model: RelocModel,
    /// Code model to use. Corresponds to `llc -code-model=$code_model`.
    /// Defaults to `None` which means "inherited from the base LLVM target".
    pub code_model: Option<CodeModel>,
    /// TLS model to use. Options are "global-dynamic" (default), "local-dynamic", "initial-exec"
    /// and "local-exec". This is similar to the -ftls-model option in GCC/Clang.
    pub tls_model: TlsModel,
    /// Do not emit code that uses the "red zone", if the ABI has one. Defaults to false.
    pub disable_redzone: bool,
    /// Frame pointer mode for this target. Defaults to `MayOmit`.
    pub frame_pointer: FramePointer,
    /// Emit each function in its own section. Defaults to true.
    pub function_sections: bool,
    /// String to prepend to the name of every dynamic library. Defaults to "lib".
    pub dll_prefix: StaticCow<str>,
    /// String to append to the name of every dynamic library. Defaults to ".so".
    pub dll_suffix: StaticCow<str>,
    /// String to append to the name of every executable.
    pub exe_suffix: StaticCow<str>,
    /// String to prepend to the name of every static library. Defaults to "lib".
    pub staticlib_prefix: StaticCow<str>,
    /// String to append to the name of every static library. Defaults to ".a".
    pub staticlib_suffix: StaticCow<str>,
    /// Values of the `target_family` cfg set for this target.
    ///
    /// Common options are: "unix", "windows". Defaults to no families.
    ///
    /// See <https://doc.rust-lang.org/reference/conditional-compilation.html#target_family>.
    pub families: StaticCow<[StaticCow<str>]>,
    /// Whether the target toolchain's ABI supports returning small structs as an integer.
    pub abi_return_struct_as_int: bool,
    /// Whether the target toolchain is like AIX's. Linker options on AIX are special and it uses
    /// XCOFF as binary format. Defaults to false.
    pub is_like_aix: bool,
    /// Whether the target toolchain is like macOS's. Only useful for compiling against iOS/macOS,
    /// in particular running dsymutil and some other stuff like `-dead_strip`. Defaults to false.
    /// Also indiates whether to use Apple-specific ABI changes, such as extending function
    /// parameters to 32-bits.
    pub is_like_osx: bool,
    /// Whether the target toolchain is like Solaris's.
    /// Only useful for compiling against Illumos/Solaris,
    /// as they have a different set of linker flags. Defaults to false.
    pub is_like_solaris: bool,
    /// Whether the target is like Windows.
    /// This is a combination of several more specific properties represented as a single flag:
    ///   - The target uses a Windows ABI,
    ///   - uses PE/COFF as a format for object code,
    ///   - uses Windows-style dllexport/dllimport for shared libraries,
    ///   - uses import libraries and .def files for symbol exports,
    ///   - executables support setting a subsystem.
    pub is_like_windows: bool,
    /// Whether the target is like MSVC.
    /// This is a combination of several more specific properties represented as a single flag:
    ///   - The target has all the properties from `is_like_windows`
    ///     (for in-tree targets "is_like_msvc ⇒ is_like_windows" is ensured by a unit test),
    ///   - has some MSVC-specific Windows ABI properties,
    ///   - uses a link.exe-like linker,
    ///   - uses CodeView/PDB for debuginfo and natvis for its visualization,
    ///   - uses SEH-based unwinding,
    ///   - supports control flow guard mechanism.
    pub is_like_msvc: bool,
    /// Whether a target toolchain is like WASM.
    pub is_like_wasm: bool,
    /// Whether a target toolchain is like Android, implying a Linux kernel and a Bionic libc
    pub is_like_android: bool,
    /// Default supported version of DWARF on this platform.
    /// Useful because some platforms (osx, bsd) only want up to DWARF2.
    pub default_dwarf_version: u32,
    /// The MinGW toolchain has a known issue that prevents it from correctly
    /// handling COFF object files with more than 2<sup>15</sup> sections. Since each weak
    /// symbol needs its own COMDAT section, weak linkage implies a large
    /// number sections that easily exceeds the given limit for larger
    /// codebases. Consequently we want a way to disallow weak linkage on some
    /// platforms.
    pub allows_weak_linkage: bool,
    /// Whether the linker support rpaths or not. Defaults to false.
    pub has_rpath: bool,
    /// Whether to disable linking to the default libraries, typically corresponds
    /// to `-nodefaultlibs`. Defaults to true.
    pub no_default_libraries: bool,
    /// Dynamically linked executables can be compiled as position independent
    /// if the default relocation model of position independent code is not
    /// changed. This is a requirement to take advantage of ASLR, as otherwise
    /// the functions in the executable are not randomized and can be used
    /// during an exploit of a vulnerability in any code.
    pub position_independent_executables: bool,
    /// Executables that are both statically linked and position-independent are supported.
    pub static_position_independent_executables: bool,
    /// Determines if the target always requires using the PLT for indirect
    /// library calls or not. This controls the default value of the `-Z plt` flag.
    pub plt_by_default: bool,
    /// Either partial, full, or off. Full RELRO makes the dynamic linker
    /// resolve all symbols at startup and marks the GOT read-only before
    /// starting the program, preventing overwriting the GOT.
    pub relro_level: RelroLevel,
    /// Format that archives should be emitted in. This affects whether we use
    /// LLVM to assemble an archive or fall back to the system linker, and
    /// currently only "gnu" is used to fall into LLVM. Unknown strings cause
    /// the system linker to be used.
    pub archive_format: StaticCow<str>,
    /// Is asm!() allowed? Defaults to true.
    pub allow_asm: bool,
    /// Whether the runtime startup code requires the `main` function be passed
    /// `argc` and `argv` values.
    pub main_needs_argc_argv: bool,

    /// Flag indicating whether #[thread_local] is available for this target.
    pub has_thread_local: bool,
    /// This is mainly for easy compatibility with emscripten.
    /// If we give emcc .o files that are actually .bc files it
    /// will 'just work'.
    pub obj_is_bitcode: bool,
    /// Whether the target requires that emitted object code includes bitcode.
    pub forces_embed_bitcode: bool,
    /// Content of the LLVM cmdline section associated with embedded bitcode.
    pub bitcode_llvm_cmdline: StaticCow<str>,

    /// Don't use this field; instead use the `.min_atomic_width()` method.
    pub min_atomic_width: Option<u64>,

    /// Don't use this field; instead use the `.max_atomic_width()` method.
    pub max_atomic_width: Option<u64>,

    /// Whether the target supports atomic CAS operations natively
    pub atomic_cas: bool,

    /// Panic strategy: "unwind" or "abort"
    pub panic_strategy: PanicStrategy,

    /// Whether or not linking dylibs to a static CRT is allowed.
    pub crt_static_allows_dylibs: bool,
    /// Whether or not the CRT is statically linked by default.
    pub crt_static_default: bool,
    /// Whether or not crt-static is respected by the compiler (or is a no-op).
    pub crt_static_respected: bool,

    /// The implementation of stack probes to use.
    pub stack_probes: StackProbeType,

    /// The minimum alignment for global symbols.
    pub min_global_align: Option<u64>,

    /// Default number of codegen units to use in debug mode
    pub default_codegen_units: Option<u64>,

    /// Whether to generate trap instructions in places where optimization would
    /// otherwise produce control flow that falls through into unrelated memory.
    pub trap_unreachable: bool,

    /// This target requires everything to be compiled with LTO to emit a final
    /// executable, aka there is no native linker for this target.
    pub requires_lto: bool,

    /// This target has no support for threads.
    pub singlethread: bool,

    /// Whether library functions call lowering/optimization is disabled in LLVM
    /// for this target unconditionally.
    pub no_builtins: bool,

    /// The default visibility for symbols in this target should be "hidden"
    /// rather than "default"
    pub default_hidden_visibility: bool,

    /// Whether a .debug_gdb_scripts section will be added to the output object file
    pub emit_debug_gdb_scripts: bool,

    /// Whether or not to unconditionally `uwtable` attributes on functions,
    /// typically because the platform needs to unwind for things like stack
    /// unwinders.
    pub requires_uwtable: bool,

    /// Whether or not to emit `uwtable` attributes on functions if `-C force-unwind-tables`
    /// is not specified and `uwtable` is not required on this target.
    pub default_uwtable: bool,

    /// Whether or not SIMD types are passed by reference in the Rust ABI,
    /// typically required if a target can be compiled with a mixed set of
    /// target features. This is `true` by default, and `false` for targets like
    /// wasm32 where the whole program either has simd or not.
    pub simd_types_indirect: bool,

    /// Pass a list of symbol which should be exported in the dylib to the linker.
    pub limit_rdylib_exports: bool,

    /// If set, have the linker export exactly these symbols, instead of using
    /// the usual logic to figure this out from the crate itself.
    pub override_export_symbols: Option<StaticCow<[StaticCow<str>]>>,

    /// Determines how or whether the MergeFunctions LLVM pass should run for
    /// this target. Either "disabled", "trampolines", or "aliases".
    /// The MergeFunctions pass is generally useful, but some targets may need
    /// to opt out. The default is "aliases".
    ///
    /// Workaround for: <https://github.com/rust-lang/rust/issues/57356>
    pub merge_functions: MergeFunctions,

    /// Use platform dependent mcount function
    pub mcount: StaticCow<str>,

    /// LLVM ABI name, corresponds to the '-mabi' parameter available in multilib C compilers
    pub llvm_abiname: StaticCow<str>,

    /// Whether or not RelaxElfRelocation flag will be passed to the linker
    pub relax_elf_relocations: bool,

    /// Additional arguments to pass to LLVM, similar to the `-C llvm-args` codegen option.
    pub llvm_args: StaticCow<[StaticCow<str>]>,

    /// Whether to use legacy .ctors initialization hooks rather than .init_array. Defaults
    /// to false (uses .init_array).
    pub use_ctors_section: bool,

    /// Whether the linker is instructed to add a `GNU_EH_FRAME` ELF header
    /// used to locate unwinding information is passed
    /// (only has effect if the linker is `ld`-like).
    pub eh_frame_header: bool,

    /// Is true if the target is an ARM architecture using thumb v1 which allows for
    /// thumb and arm interworking.
    pub has_thumb_interworking: bool,

    /// Which kind of debuginfo is used by this target?
    pub debuginfo_kind: DebuginfoKind,
    /// How to handle split debug information, if at all. Specifying `None` has
    /// target-specific meaning.
    pub split_debuginfo: SplitDebuginfo,
    /// Which kinds of split debuginfo are supported by the target?
    pub supported_split_debuginfo: StaticCow<[SplitDebuginfo]>,

    /// The sanitizers supported by this target
    ///
    /// Note that the support here is at a codegen level. If the machine code with sanitizer
    /// enabled can generated on this target, but the necessary supporting libraries are not
    /// distributed with the target, the sanitizer should still appear in this list for the target.
    pub supported_sanitizers: SanitizerSet,

    /// If present it's a default value to use for adjusting the C ABI.
    pub default_adjusted_cabi: Option<Abi>,

    /// Minimum number of bits in #[repr(C)] enum. Defaults to the size of c_int
    pub c_enum_min_bits: Option<u64>,

    /// Whether or not the DWARF `.debug_aranges` section should be generated.
    pub generate_arange_section: bool,

    /// Whether the target supports stack canary checks. `true` by default,
    /// since this is most common among tier 1 and tier 2 targets.
    pub supports_stack_protector: bool,

    /// The name of entry function.
    /// Default value is "main"
    pub entry_name: StaticCow<str>,

    /// The ABI of entry function.
    /// Default value is `Conv::C`, i.e. C call convention
    pub entry_abi: Conv,

    /// Whether the target supports XRay instrumentation.
    pub supports_xray: bool,

    /// Forces the use of emulated TLS (__emutls_get_address)
    pub force_emulated_tls: bool,
}

/// Add arguments for the given flavor and also for its "twin" flavors
/// that have a compatible command line interface.
fn add_link_args_iter(
    link_args: &mut LinkArgs,
    flavor: LinkerFlavor,
    args: impl Iterator<Item = StaticCow<str>> + Clone,
) {
    let mut insert = |flavor| link_args.entry(flavor).or_default().extend(args.clone());
    insert(flavor);
    match flavor {
        LinkerFlavor::Gnu(cc, lld) => {
            assert_eq!(lld, Lld::No);
            insert(LinkerFlavor::Gnu(cc, Lld::Yes));
        }
        LinkerFlavor::Darwin(cc, lld) => {
            assert_eq!(lld, Lld::No);
            insert(LinkerFlavor::Darwin(cc, Lld::Yes));
        }
        LinkerFlavor::Msvc(lld) => {
            assert_eq!(lld, Lld::No);
            insert(LinkerFlavor::Msvc(Lld::Yes));
        }
        LinkerFlavor::WasmLld(..)
        | LinkerFlavor::Unix(..)
        | LinkerFlavor::EmCc
        | LinkerFlavor::Bpf
        | LinkerFlavor::Ptx => {}
    }
}

fn add_link_args(link_args: &mut LinkArgs, flavor: LinkerFlavor, args: &[&'static str]) {
    add_link_args_iter(link_args, flavor, args.iter().copied().map(Cow::Borrowed))
}

impl TargetOptions {
    fn link_args(flavor: LinkerFlavor, args: &[&'static str]) -> LinkArgs {
        let mut link_args = LinkArgs::new();
        add_link_args(&mut link_args, flavor, args);
        link_args
    }

    fn add_pre_link_args(&mut self, flavor: LinkerFlavor, args: &[&'static str]) {
        add_link_args(&mut self.pre_link_args, flavor, args);
    }

    fn add_post_link_args(&mut self, flavor: LinkerFlavor, args: &[&'static str]) {
        add_link_args(&mut self.post_link_args, flavor, args);
    }

    fn update_from_cli(&mut self) {
        self.linker_flavor = LinkerFlavor::from_cli_json(
            self.linker_flavor_json,
            self.lld_flavor_json,
            self.linker_is_gnu_json,
        );
        for (args, args_json) in [
            (&mut self.pre_link_args, &self.pre_link_args_json),
            (&mut self.late_link_args, &self.late_link_args_json),
            (&mut self.late_link_args_dynamic, &self.late_link_args_dynamic_json),
            (&mut self.late_link_args_static, &self.late_link_args_static_json),
            (&mut self.post_link_args, &self.post_link_args_json),
        ] {
            args.clear();
            for (flavor, args_json) in args_json {
                let linker_flavor = self.linker_flavor.with_cli_hints(*flavor);
                // Normalize to no lld to avoid asserts.
                let linker_flavor = match linker_flavor {
                    LinkerFlavor::Gnu(cc, _) => LinkerFlavor::Gnu(cc, Lld::No),
                    LinkerFlavor::Darwin(cc, _) => LinkerFlavor::Darwin(cc, Lld::No),
                    LinkerFlavor::Msvc(_) => LinkerFlavor::Msvc(Lld::No),
                    _ => linker_flavor,
                };
                if !args.contains_key(&linker_flavor) {
                    add_link_args_iter(args, linker_flavor, args_json.iter().cloned());
                }
            }
        }
    }

    fn update_to_cli(&mut self) {
        self.linker_flavor_json = self.linker_flavor.to_cli();
        self.lld_flavor_json = self.linker_flavor.lld_flavor();
        self.linker_is_gnu_json = self.linker_flavor.is_gnu();
        for (args, args_json) in [
            (&self.pre_link_args, &mut self.pre_link_args_json),
            (&self.late_link_args, &mut self.late_link_args_json),
            (&self.late_link_args_dynamic, &mut self.late_link_args_dynamic_json),
            (&self.late_link_args_static, &mut self.late_link_args_static_json),
            (&self.post_link_args, &mut self.post_link_args_json),
        ] {
            *args_json =
                args.iter().map(|(flavor, args)| (flavor.to_cli(), args.clone())).collect();
        }
    }
}

impl Default for TargetOptions {
    /// Creates a set of "sane defaults" for any target. This is still
    /// incomplete, and if used for compilation, will certainly not work.
    fn default() -> TargetOptions {
        TargetOptions {
            is_builtin: false,
            endian: Endian::Little,
            c_int_width: "32".into(),
            os: "none".into(),
            env: "".into(),
            abi: "".into(),
            vendor: "unknown".into(),
            linker: option_env!("CFG_DEFAULT_LINKER").map(|s| s.into()),
            linker_flavor: LinkerFlavor::Gnu(Cc::Yes, Lld::No),
            linker_flavor_json: LinkerFlavorCli::Gcc,
            lld_flavor_json: LldFlavor::Ld,
            linker_is_gnu_json: true,
            link_script: None,
            asm_args: cvs![],
            cpu: "generic".into(),
            features: "".into(),
            dynamic_linking: false,
            dll_tls_export: true,
            only_cdylib: false,
            executables: true,
            relocation_model: RelocModel::Pic,
            code_model: None,
            tls_model: TlsModel::GeneralDynamic,
            disable_redzone: false,
            frame_pointer: FramePointer::MayOmit,
            function_sections: true,
            dll_prefix: "lib".into(),
            dll_suffix: ".so".into(),
            exe_suffix: "".into(),
            staticlib_prefix: "lib".into(),
            staticlib_suffix: ".a".into(),
            families: cvs![],
            abi_return_struct_as_int: false,
            is_like_aix: false,
            is_like_osx: false,
            is_like_solaris: false,
            is_like_windows: false,
            is_like_msvc: false,
            is_like_wasm: false,
            is_like_android: false,
            default_dwarf_version: 4,
            allows_weak_linkage: true,
            has_rpath: false,
            no_default_libraries: true,
            position_independent_executables: false,
            static_position_independent_executables: false,
            plt_by_default: true,
            relro_level: RelroLevel::None,
            pre_link_objects: Default::default(),
            post_link_objects: Default::default(),
            pre_link_objects_self_contained: Default::default(),
            post_link_objects_self_contained: Default::default(),
            link_self_contained: LinkSelfContainedDefault::False,
            pre_link_args: LinkArgs::new(),
            pre_link_args_json: LinkArgsCli::new(),
            late_link_args: LinkArgs::new(),
            late_link_args_json: LinkArgsCli::new(),
            late_link_args_dynamic: LinkArgs::new(),
            late_link_args_dynamic_json: LinkArgsCli::new(),
            late_link_args_static: LinkArgs::new(),
            late_link_args_static_json: LinkArgsCli::new(),
            post_link_args: LinkArgs::new(),
            post_link_args_json: LinkArgsCli::new(),
            link_env: cvs![],
            link_env_remove: cvs![],
            archive_format: "gnu".into(),
            main_needs_argc_argv: true,
            allow_asm: true,
            has_thread_local: false,
            obj_is_bitcode: false,
            forces_embed_bitcode: false,
            bitcode_llvm_cmdline: "".into(),
            min_atomic_width: None,
            max_atomic_width: None,
            atomic_cas: true,
            panic_strategy: PanicStrategy::Unwind,
            crt_static_allows_dylibs: false,
            crt_static_default: false,
            crt_static_respected: false,
            stack_probes: StackProbeType::None,
            min_global_align: None,
            default_codegen_units: None,
            trap_unreachable: true,
            requires_lto: false,
            singlethread: false,
            no_builtins: false,
            default_hidden_visibility: false,
            emit_debug_gdb_scripts: true,
            requires_uwtable: false,
            default_uwtable: false,
            simd_types_indirect: true,
            limit_rdylib_exports: true,
            override_export_symbols: None,
            merge_functions: MergeFunctions::Aliases,
            mcount: "mcount".into(),
            llvm_abiname: "".into(),
            relax_elf_relocations: false,
            llvm_args: cvs![],
            use_ctors_section: false,
            eh_frame_header: true,
            has_thumb_interworking: false,
            debuginfo_kind: Default::default(),
            split_debuginfo: Default::default(),
            // `Off` is supported by default, but targets can remove this manually, e.g. Windows.
            supported_split_debuginfo: Cow::Borrowed(&[SplitDebuginfo::Off]),
            supported_sanitizers: SanitizerSet::empty(),
            default_adjusted_cabi: None,
            c_enum_min_bits: None,
            generate_arange_section: true,
            supports_stack_protector: true,
            entry_name: "main".into(),
            entry_abi: Conv::C,
            supports_xray: false,
            force_emulated_tls: false,
        }
    }
}

/// `TargetOptions` being a separate type is basically an implementation detail of `Target` that is
/// used for providing defaults. Perhaps there's a way to merge `TargetOptions` into `Target` so
/// this `Deref` implementation is no longer necessary.
impl Deref for Target {
    type Target = TargetOptions;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.options
    }
}
impl DerefMut for Target {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.options
    }
}

impl Target {
    /// Given a function ABI, turn it into the correct ABI for this target.
    pub fn adjust_abi(&self, abi: Abi) -> Abi {
        match abi {
            Abi::C { .. } => self.default_adjusted_cabi.unwrap_or(abi),
            Abi::System { unwind } if self.is_like_windows && self.arch == "x86" => {
                Abi::Stdcall { unwind }
            }
            Abi::System { unwind } => Abi::C { unwind },
            Abi::EfiApi if self.arch == "arm" => Abi::Aapcs { unwind: false },
            Abi::EfiApi if self.arch == "x86_64" => Abi::Win64 { unwind: false },
            Abi::EfiApi => Abi::C { unwind: false },

            // See commentary in `is_abi_supported`.
            Abi::Stdcall { .. } | Abi::Thiscall { .. } if self.arch == "x86" => abi,
            Abi::Stdcall { unwind } | Abi::Thiscall { unwind } => Abi::C { unwind },
            Abi::Fastcall { .. } if self.arch == "x86" => abi,
            Abi::Vectorcall { .. } if ["x86", "x86_64"].contains(&&self.arch[..]) => abi,
            Abi::Fastcall { unwind } | Abi::Vectorcall { unwind } => Abi::C { unwind },

            abi => abi,
        }
    }

    /// Returns a None if the UNSUPPORTED_CALLING_CONVENTIONS lint should be emitted
    pub fn is_abi_supported(&self, abi: Abi) -> Option<bool> {
        use Abi::*;
        Some(match abi {
            Rust
            | C { .. }
            | System { .. }
            | RustIntrinsic
            | RustCall
            | PlatformIntrinsic
            | Unadjusted
            | Cdecl { .. }
            | RustCold => true,
            EfiApi => {
                ["arm", "aarch64", "riscv32", "riscv64", "x86", "x86_64"].contains(&&self.arch[..])
            }
            X86Interrupt => ["x86", "x86_64"].contains(&&self.arch[..]),
            Aapcs { .. } => "arm" == self.arch,
            CCmseNonSecureCall => ["arm", "aarch64"].contains(&&self.arch[..]),
            Win64 { .. } | SysV64 { .. } => self.arch == "x86_64",
            PtxKernel => self.arch == "nvptx64",
            Msp430Interrupt => self.arch == "msp430",
            AmdGpuKernel => self.arch == "amdgcn",
            AvrInterrupt | AvrNonBlockingInterrupt => self.arch == "avr",
            Wasm => ["wasm32", "wasm64"].contains(&&self.arch[..]),
            Thiscall { .. } => self.arch == "x86",
            // On windows these fall-back to platform native calling convention (C) when the
            // architecture is not supported.
            //
            // This is I believe a historical accident that has occurred as part of Microsoft
            // striving to allow most of the code to "just" compile when support for 64-bit x86
            // was added and then later again, when support for ARM architectures was added.
            //
            // This is well documented across MSDN. Support for this in Rust has been added in
            // #54576. This makes much more sense in context of Microsoft's C++ than it does in
            // Rust, but there isn't much leeway remaining here to change it back at the time this
            // comment has been written.
            //
            // Following are the relevant excerpts from the MSDN documentation.
            //
            // > The __vectorcall calling convention is only supported in native code on x86 and
            // x64 processors that include Streaming SIMD Extensions 2 (SSE2) and above.
            // > ...
            // > On ARM machines, __vectorcall is accepted and ignored by the compiler.
            //
            // -- https://docs.microsoft.com/en-us/cpp/cpp/vectorcall?view=msvc-160
            //
            // > On ARM and x64 processors, __stdcall is accepted and ignored by the compiler;
            //
            // -- https://docs.microsoft.com/en-us/cpp/cpp/stdcall?view=msvc-160
            //
            // > In most cases, keywords or compiler switches that specify an unsupported
            // > convention on a particular platform are ignored, and the platform default
            // > convention is used.
            //
            // -- https://docs.microsoft.com/en-us/cpp/cpp/argument-passing-and-naming-conventions
            Stdcall { .. } | Fastcall { .. } | Vectorcall { .. } if self.is_like_windows => true,
            // Outside of Windows we want to only support these calling conventions for the
            // architectures for which these calling conventions are actually well defined.
            Stdcall { .. } | Fastcall { .. } if self.arch == "x86" => true,
            Vectorcall { .. } if ["x86", "x86_64"].contains(&&self.arch[..]) => true,
            // Return a `None` for other cases so that we know to emit a future compat lint.
            Stdcall { .. } | Fastcall { .. } | Vectorcall { .. } => return None,
        })
    }

    /// Minimum integer size in bits that this target can perform atomic
    /// operations on.
    pub fn min_atomic_width(&self) -> u64 {
        self.min_atomic_width.unwrap_or(8)
    }

    /// Maximum integer size in bits that this target can perform atomic
    /// operations on.
    pub fn max_atomic_width(&self) -> u64 {
        self.max_atomic_width.unwrap_or_else(|| self.pointer_width.into())
    }

    /// Loads a target descriptor from a JSON object.
    pub fn from_json(obj: Json) -> Result<(Target, TargetWarnings), String> {
        // While ugly, this code must remain this way to retain
        // compatibility with existing JSON fields and the internal
        // expected naming of the Target and TargetOptions structs.
        // To ensure compatibility is retained, the built-in targets
        // are round-tripped through this code to catch cases where
        // the JSON parser is not updated to match the structs.

        let mut obj = match obj {
            Value::Object(obj) => obj,
            _ => return Err("Expected JSON object for target")?,
        };

        let mut get_req_field = |name: &str| {
            obj.remove(name)
                .and_then(|j| j.as_str().map(str::to_string))
                .ok_or_else(|| format!("Field {name} in target specification is required"))
        };

        let mut base = Target {
            llvm_target: get_req_field("llvm-target")?.into(),
            pointer_width: get_req_field("target-pointer-width")?
                .parse::<u32>()
                .map_err(|_| "target-pointer-width must be an integer".to_string())?,
            data_layout: get_req_field("data-layout")?.into(),
            arch: get_req_field("arch")?.into(),
            options: Default::default(),
        };

        let mut incorrect_type = vec![];

        macro_rules! key {
            ($key_name:ident) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(s) = obj.remove(&name).and_then(|s| s.as_str().map(str::to_string).map(Cow::from)) {
                    base.$key_name = s;
                }
            } );
            ($key_name:ident = $json_name:expr) => ( {
                let name = $json_name;
                if let Some(s) = obj.remove(name).and_then(|s| s.as_str().map(str::to_string).map(Cow::from)) {
                    base.$key_name = s;
                }
            } );
            ($key_name:ident, bool) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(s) = obj.remove(&name).and_then(|b| b.as_bool()) {
                    base.$key_name = s;
                }
            } );
            ($key_name:ident = $json_name:expr, bool) => ( {
                let name = $json_name;
                if let Some(s) = obj.remove(name).and_then(|b| b.as_bool()) {
                    base.$key_name = s;
                }
            } );
            ($key_name:ident, u32) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(s) = obj.remove(&name).and_then(|b| b.as_u64()) {
                    if s < 1 || s > 5 {
                        return Err("Not a valid DWARF version number".into());
                    }
                    base.$key_name = s as u32;
                }
            } );
            ($key_name:ident, Option<u64>) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(s) = obj.remove(&name).and_then(|b| b.as_u64()) {
                    base.$key_name = Some(s);
                }
            } );
            ($key_name:ident, MergeFunctions) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<MergeFunctions>() {
                        Ok(mergefunc) => base.$key_name = mergefunc,
                        _ => return Some(Err(format!("'{}' is not a valid value for \
                                                      merge-functions. Use 'disabled', \
                                                      'trampolines', or 'aliases'.",
                                                      s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, RelocModel) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<RelocModel>() {
                        Ok(relocation_model) => base.$key_name = relocation_model,
                        _ => return Some(Err(format!("'{}' is not a valid relocation model. \
                                                      Run `rustc --print relocation-models` to \
                                                      see the list of supported values.", s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, CodeModel) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<CodeModel>() {
                        Ok(code_model) => base.$key_name = Some(code_model),
                        _ => return Some(Err(format!("'{}' is not a valid code model. \
                                                      Run `rustc --print code-models` to \
                                                      see the list of supported values.", s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, TlsModel) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<TlsModel>() {
                        Ok(tls_model) => base.$key_name = tls_model,
                        _ => return Some(Err(format!("'{}' is not a valid TLS model. \
                                                      Run `rustc --print tls-models` to \
                                                      see the list of supported values.", s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, PanicStrategy) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s {
                        "unwind" => base.$key_name = PanicStrategy::Unwind,
                        "abort" => base.$key_name = PanicStrategy::Abort,
                        _ => return Some(Err(format!("'{}' is not a valid value for \
                                                      panic-strategy. Use 'unwind' or 'abort'.",
                                                     s))),
                }
                Some(Ok(()))
            })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, RelroLevel) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<RelroLevel>() {
                        Ok(level) => base.$key_name = level,
                        _ => return Some(Err(format!("'{}' is not a valid value for \
                                                      relro-level. Use 'full', 'partial, or 'off'.",
                                                      s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, DebuginfoKind) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<DebuginfoKind>() {
                        Ok(level) => base.$key_name = level,
                        _ => return Some(Err(
                            format!("'{s}' is not a valid value for debuginfo-kind. Use 'dwarf', \
                                  'dwarf-dsym' or 'pdb'.")
                        )),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, SplitDebuginfo) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<SplitDebuginfo>() {
                        Ok(level) => base.$key_name = level,
                        _ => return Some(Err(format!("'{}' is not a valid value for \
                                                      split-debuginfo. Use 'off' or 'dsymutil'.",
                                                      s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, list) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(j) = obj.remove(&name) {
                    if let Some(v) = j.as_array() {
                        base.$key_name = v.iter()
                            .map(|a| a.as_str().unwrap().to_string().into())
                            .collect();
                    } else {
                        incorrect_type.push(name)
                    }
                }
            } );
            ($key_name:ident, opt_list) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(j) = obj.remove(&name) {
                    if let Some(v) = j.as_array() {
                        base.$key_name = Some(v.iter()
                            .map(|a| a.as_str().unwrap().to_string().into())
                            .collect());
                    } else {
                        incorrect_type.push(name)
                    }
                }
            } );
            ($key_name:ident, fallible_list) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|j| {
                    if let Some(v) = j.as_array() {
                        match v.iter().map(|a| FromStr::from_str(a.as_str().unwrap())).collect() {
                            Ok(l) => { base.$key_name = l },
                            // FIXME: `fallible_list` can't re-use the `key!` macro for list
                            // elements and the error messages from that macro, so it has a bad
                            // generic message instead
                            Err(_) => return Some(Err(
                                format!("`{:?}` is not a valid value for `{}`", j, name)
                            )),
                        }
                    } else {
                        incorrect_type.push(name)
                    }
                    Some(Ok(()))
                }).unwrap_or(Ok(()))
            } );
            ($key_name:ident, optional) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(o) = obj.remove(&name) {
                    base.$key_name = o
                        .as_str()
                        .map(|s| s.to_string().into());
                }
            } );
            ($key_name:ident = $json_name:expr, LldFlavor) => ( {
                let name = $json_name;
                obj.remove(name).and_then(|o| o.as_str().and_then(|s| {
                    if let Some(flavor) = LldFlavor::from_str(&s) {
                        base.$key_name = flavor;
                    } else {
                        return Some(Err(format!(
                            "'{}' is not a valid value for lld-flavor. \
                             Use 'darwin', 'gnu', 'link' or 'wasm'.",
                            s)))
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident = $json_name:expr, LinkerFlavor) => ( {
                let name = $json_name;
                obj.remove(name).and_then(|o| o.as_str().and_then(|s| {
                    match LinkerFlavorCli::from_str(s) {
                        Some(linker_flavor) => base.$key_name = linker_flavor,
                        _ => return Some(Err(format!("'{}' is not a valid value for linker-flavor. \
                                                      Use {}", s, LinkerFlavorCli::one_of()))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, StackProbeType) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| match StackProbeType::from_json(&o) {
                    Ok(v) => {
                        base.$key_name = v;
                        Some(Ok(()))
                    },
                    Err(s) => Some(Err(
                        format!("`{:?}` is not a valid value for `{}`: {}", o, name, s)
                    )),
                }).unwrap_or(Ok(()))
            } );
            ($key_name:ident, SanitizerSet) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(o) = obj.remove(&name) {
                    if let Some(a) = o.as_array() {
                        for s in a {
                            base.$key_name |= match s.as_str() {
                                Some("address") => SanitizerSet::ADDRESS,
                                Some("cfi") => SanitizerSet::CFI,
                                Some("kcfi") => SanitizerSet::KCFI,
                                Some("kernel-address") => SanitizerSet::KERNELADDRESS,
                                Some("leak") => SanitizerSet::LEAK,
                                Some("memory") => SanitizerSet::MEMORY,
                                Some("memtag") => SanitizerSet::MEMTAG,
                                Some("safestack") => SanitizerSet::SAFESTACK,
                                Some("shadow-call-stack") => SanitizerSet::SHADOWCALLSTACK,
                                Some("thread") => SanitizerSet::THREAD,
                                Some("hwaddress") => SanitizerSet::HWADDRESS,
                                Some(s) => return Err(format!("unknown sanitizer {}", s)),
                                _ => return Err(format!("not a string: {:?}", s)),
                            };
                        }
                    } else {
                        incorrect_type.push(name)
                    }
                }
                Ok::<(), String>(())
            } );

            ($key_name:ident = $json_name:expr, link_self_contained) => ( {
                let name = $json_name;
                obj.remove(name).and_then(|o| o.as_str().and_then(|s| {
                    match s.parse::<LinkSelfContainedDefault>() {
                        Ok(lsc_default) => base.$key_name = lsc_default,
                        _ => return Some(Err(format!("'{}' is not a valid `-Clink-self-contained` default. \
                                                      Use 'false', 'true', 'musl' or 'mingw'", s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident = $json_name:expr, link_objects) => ( {
                let name = $json_name;
                if let Some(val) = obj.remove(name) {
                    let obj = val.as_object().ok_or_else(|| format!("{}: expected a \
                        JSON object with fields per CRT object kind.", name))?;
                    let mut args = CrtObjects::new();
                    for (k, v) in obj {
                        let kind = LinkOutputKind::from_str(&k).ok_or_else(|| {
                            format!("{}: '{}' is not a valid value for CRT object kind. \
                                     Use '(dynamic,static)-(nopic,pic)-exe' or \
                                     '(dynamic,static)-dylib' or 'wasi-reactor-exe'", name, k)
                        })?;

                        let v = v.as_array().ok_or_else(||
                            format!("{}.{}: expected a JSON array", name, k)
                        )?.iter().enumerate()
                            .map(|(i,s)| {
                                let s = s.as_str().ok_or_else(||
                                    format!("{}.{}[{}]: expected a JSON string", name, k, i))?;
                                Ok(s.to_string().into())
                            })
                            .collect::<Result<Vec<_>, String>>()?;

                        args.insert(kind, v);
                    }
                    base.$key_name = args;
                }
            } );
            ($key_name:ident = $json_name:expr, link_args) => ( {
                let name = $json_name;
                if let Some(val) = obj.remove(name) {
                    let obj = val.as_object().ok_or_else(|| format!("{}: expected a \
                        JSON object with fields per linker-flavor.", name))?;
                    let mut args = LinkArgsCli::new();
                    for (k, v) in obj {
                        let flavor = LinkerFlavorCli::from_str(&k).ok_or_else(|| {
                            format!("{}: '{}' is not a valid value for linker-flavor. \
                                     Use 'em', 'gcc', 'ld' or 'msvc'", name, k)
                        })?;

                        let v = v.as_array().ok_or_else(||
                            format!("{}.{}: expected a JSON array", name, k)
                        )?.iter().enumerate()
                            .map(|(i,s)| {
                                let s = s.as_str().ok_or_else(||
                                    format!("{}.{}[{}]: expected a JSON string", name, k, i))?;
                                Ok(s.to_string().into())
                            })
                            .collect::<Result<Vec<_>, String>>()?;

                        args.insert(flavor, v);
                    }
                    base.$key_name = args;
                }
            } );
            ($key_name:ident, env) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(o) = obj.remove(&name) {
                    if let Some(a) = o.as_array() {
                        for o in a {
                            if let Some(s) = o.as_str() {
                                let p = s.split('=').collect::<Vec<_>>();
                                if p.len() == 2 {
                                    let k = p[0].to_string();
                                    let v = p[1].to_string();
                                    base.$key_name.to_mut().push((k.into(), v.into()));
                                }
                            }
                        }
                    } else {
                        incorrect_type.push(name)
                    }
                }
            } );
            ($key_name:ident, Option<Abi>) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match lookup_abi(s) {
                        Some(abi) => base.$key_name = Some(abi),
                        _ => return Some(Err(format!("'{}' is not a valid value for abi", s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, TargetFamilies) => ( {
                if let Some(value) = obj.remove("target-family") {
                    if let Some(v) = value.as_array() {
                        base.$key_name = v.iter()
                            .map(|a| a.as_str().unwrap().to_string().into())
                            .collect();
                    } else if let Some(v) = value.as_str() {
                        base.$key_name = vec![v.to_string().into()].into();
                    }
                }
            } );
            ($key_name:ident, Conv) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.remove(&name).and_then(|o| o.as_str().and_then(|s| {
                    match Conv::from_str(s) {
                        Ok(c) => {
                            base.$key_name = c;
                            Some(Ok(()))
                        }
                        Err(e) => Some(Err(e))
                    }
                })).unwrap_or(Ok(()))
            } );
        }

        if let Some(j) = obj.remove("target-endian") {
            if let Some(s) = j.as_str() {
                base.endian = s.parse()?;
            } else {
                incorrect_type.push("target-endian".into())
            }
        }

        if let Some(fp) = obj.remove("frame-pointer") {
            if let Some(s) = fp.as_str() {
                base.frame_pointer = s
                    .parse()
                    .map_err(|()| format!("'{s}' is not a valid value for frame-pointer"))?;
            } else {
                incorrect_type.push("frame-pointer".into())
            }
        }

        key!(is_builtin, bool);
        key!(c_int_width = "target-c-int-width");
        key!(c_enum_min_bits, Option<u64>); // if None, matches c_int_width
        key!(os);
        key!(env);
        key!(abi);
        key!(vendor);
        key!(linker, optional);
        key!(linker_flavor_json = "linker-flavor", LinkerFlavor)?;
        key!(lld_flavor_json = "lld-flavor", LldFlavor)?;
        key!(linker_is_gnu_json = "linker-is-gnu", bool);
        key!(pre_link_objects = "pre-link-objects", link_objects);
        key!(post_link_objects = "post-link-objects", link_objects);
        key!(pre_link_objects_self_contained = "pre-link-objects-fallback", link_objects);
        key!(post_link_objects_self_contained = "post-link-objects-fallback", link_objects);
        key!(link_self_contained = "crt-objects-fallback", link_self_contained)?;
        key!(pre_link_args_json = "pre-link-args", link_args);
        key!(late_link_args_json = "late-link-args", link_args);
        key!(late_link_args_dynamic_json = "late-link-args-dynamic", link_args);
        key!(late_link_args_static_json = "late-link-args-static", link_args);
        key!(post_link_args_json = "post-link-args", link_args);
        key!(link_script, optional);
        key!(link_env, env);
        key!(link_env_remove, list);
        key!(asm_args, list);
        key!(cpu);
        key!(features);
        key!(dynamic_linking, bool);
        key!(dll_tls_export, bool);
        key!(only_cdylib, bool);
        key!(executables, bool);
        key!(relocation_model, RelocModel)?;
        key!(code_model, CodeModel)?;
        key!(tls_model, TlsModel)?;
        key!(disable_redzone, bool);
        key!(function_sections, bool);
        key!(dll_prefix);
        key!(dll_suffix);
        key!(exe_suffix);
        key!(staticlib_prefix);
        key!(staticlib_suffix);
        key!(families, TargetFamilies);
        key!(abi_return_struct_as_int, bool);
        key!(is_like_aix, bool);
        key!(is_like_osx, bool);
        key!(is_like_solaris, bool);
        key!(is_like_windows, bool);
        key!(is_like_msvc, bool);
        key!(is_like_wasm, bool);
        key!(is_like_android, bool);
        key!(default_dwarf_version, u32);
        key!(allows_weak_linkage, bool);
        key!(has_rpath, bool);
        key!(no_default_libraries, bool);
        key!(position_independent_executables, bool);
        key!(static_position_independent_executables, bool);
        key!(plt_by_default, bool);
        key!(relro_level, RelroLevel)?;
        key!(archive_format);
        key!(allow_asm, bool);
        key!(main_needs_argc_argv, bool);
        key!(has_thread_local, bool);
        key!(obj_is_bitcode, bool);
        key!(forces_embed_bitcode, bool);
        key!(bitcode_llvm_cmdline);
        key!(max_atomic_width, Option<u64>);
        key!(min_atomic_width, Option<u64>);
        key!(atomic_cas, bool);
        key!(panic_strategy, PanicStrategy)?;
        key!(crt_static_allows_dylibs, bool);
        key!(crt_static_default, bool);
        key!(crt_static_respected, bool);
        key!(stack_probes, StackProbeType)?;
        key!(min_global_align, Option<u64>);
        key!(default_codegen_units, Option<u64>);
        key!(trap_unreachable, bool);
        key!(requires_lto, bool);
        key!(singlethread, bool);
        key!(no_builtins, bool);
        key!(default_hidden_visibility, bool);
        key!(emit_debug_gdb_scripts, bool);
        key!(requires_uwtable, bool);
        key!(default_uwtable, bool);
        key!(simd_types_indirect, bool);
        key!(limit_rdylib_exports, bool);
        key!(override_export_symbols, opt_list);
        key!(merge_functions, MergeFunctions)?;
        key!(mcount = "target-mcount");
        key!(llvm_abiname);
        key!(relax_elf_relocations, bool);
        key!(llvm_args, list);
        key!(use_ctors_section, bool);
        key!(eh_frame_header, bool);
        key!(has_thumb_interworking, bool);
        key!(debuginfo_kind, DebuginfoKind)?;
        key!(split_debuginfo, SplitDebuginfo)?;
        key!(supported_split_debuginfo, fallible_list)?;
        key!(supported_sanitizers, SanitizerSet)?;
        key!(default_adjusted_cabi, Option<Abi>)?;
        key!(generate_arange_section, bool);
        key!(supports_stack_protector, bool);
        key!(entry_name);
        key!(entry_abi, Conv)?;
        key!(supports_xray, bool);
        key!(force_emulated_tls, bool);

        if base.is_builtin {
            // This can cause unfortunate ICEs later down the line.
            return Err("may not set is_builtin for targets not built-in".into());
        }
        base.update_from_cli();

        // Each field should have been read using `Json::remove` so any keys remaining are unused.
        let remaining_keys = obj.keys();
        Ok((
            base,
            TargetWarnings { unused_fields: remaining_keys.cloned().collect(), incorrect_type },
        ))
    }

    /// Load a built-in target
    pub fn expect_builtin(target_triple: &TargetTriple) -> Target {
        match *target_triple {
            TargetTriple::TargetTriple(ref target_triple) => {
                load_builtin(target_triple).expect("built-in target")
            }
            TargetTriple::TargetJson { .. } => {
                panic!("built-in targets doesn't support target-paths")
            }
        }
    }

    /// Search for a JSON file specifying the given target triple.
    ///
    /// If none is found in `$RUST_TARGET_PATH`, look for a file called `target.json` inside the
    /// sysroot under the target-triple's `rustlib` directory. Note that it could also just be a
    /// bare filename already, so also check for that. If one of the hardcoded targets we know
    /// about, just return it directly.
    ///
    /// The error string could come from any of the APIs called, including filesystem access and
    /// JSON decoding.
    pub fn search(
        target_triple: &TargetTriple,
        sysroot: &Path,
    ) -> Result<(Target, TargetWarnings), String> {
        use std::env;
        use std::fs;

        fn load_file(path: &Path) -> Result<(Target, TargetWarnings), String> {
            let contents = fs::read_to_string(path).map_err(|e| e.to_string())?;
            let obj = serde_json::from_str(&contents).map_err(|e| e.to_string())?;
            Target::from_json(obj)
        }

        match *target_triple {
            TargetTriple::TargetTriple(ref target_triple) => {
                // check if triple is in list of built-in targets
                if let Some(t) = load_builtin(target_triple) {
                    return Ok((t, TargetWarnings::empty()));
                }

                // search for a file named `target_triple`.json in RUST_TARGET_PATH
                let path = {
                    let mut target = target_triple.to_string();
                    target.push_str(".json");
                    PathBuf::from(target)
                };

                let target_path = env::var_os("RUST_TARGET_PATH").unwrap_or_default();

                for dir in env::split_paths(&target_path) {
                    let p = dir.join(&path);
                    if p.is_file() {
                        return load_file(&p);
                    }
                }

                // Additionally look in the sysroot under `lib/rustlib/<triple>/target.json`
                // as a fallback.
                let rustlib_path = crate::target_rustlib_path(sysroot, target_triple);
                let p = PathBuf::from_iter([
                    Path::new(sysroot),
                    Path::new(&rustlib_path),
                    Path::new("target.json"),
                ]);
                if p.is_file() {
                    return load_file(&p);
                }

                Err(format!("Could not find specification for target {target_triple:?}"))
            }
            TargetTriple::TargetJson { ref contents, .. } => {
                let obj = serde_json::from_str(contents).map_err(|e| e.to_string())?;
                Target::from_json(obj)
            }
        }
    }
}

impl ToJson for Target {
    fn to_json(&self) -> Json {
        let mut d = serde_json::Map::new();
        let default: TargetOptions = Default::default();
        let mut target = self.clone();
        target.update_to_cli();

        macro_rules! target_val {
            ($attr:ident) => {{
                let name = (stringify!($attr)).replace("_", "-");
                d.insert(name, target.$attr.to_json());
            }};
        }

        macro_rules! target_option_val {
            ($attr:ident) => {{
                let name = (stringify!($attr)).replace("_", "-");
                if default.$attr != target.$attr {
                    d.insert(name, target.$attr.to_json());
                }
            }};
            ($attr:ident, $json_name:expr) => {{
                let name = $json_name;
                if default.$attr != target.$attr {
                    d.insert(name.into(), target.$attr.to_json());
                }
            }};
            (link_args - $attr:ident, $json_name:expr) => {{
                let name = $json_name;
                if default.$attr != target.$attr {
                    let obj = target
                        .$attr
                        .iter()
                        .map(|(k, v)| (k.desc().to_string(), v.clone()))
                        .collect::<BTreeMap<_, _>>();
                    d.insert(name.to_string(), obj.to_json());
                }
            }};
            (env - $attr:ident) => {{
                let name = (stringify!($attr)).replace("_", "-");
                if default.$attr != target.$attr {
                    let obj = target
                        .$attr
                        .iter()
                        .map(|&(ref k, ref v)| format!("{k}={v}"))
                        .collect::<Vec<_>>();
                    d.insert(name, obj.to_json());
                }
            }};
        }

        target_val!(llvm_target);
        d.insert("target-pointer-width".to_string(), self.pointer_width.to_string().to_json());
        target_val!(arch);
        target_val!(data_layout);

        target_option_val!(is_builtin);
        target_option_val!(endian, "target-endian");
        target_option_val!(c_int_width, "target-c-int-width");
        target_option_val!(os);
        target_option_val!(env);
        target_option_val!(abi);
        target_option_val!(vendor);
        target_option_val!(linker);
        target_option_val!(linker_flavor_json, "linker-flavor");
        target_option_val!(lld_flavor_json, "lld-flavor");
        target_option_val!(linker_is_gnu_json, "linker-is-gnu");
        target_option_val!(pre_link_objects);
        target_option_val!(post_link_objects);
        target_option_val!(pre_link_objects_self_contained, "pre-link-objects-fallback");
        target_option_val!(post_link_objects_self_contained, "post-link-objects-fallback");
        target_option_val!(link_self_contained, "crt-objects-fallback");
        target_option_val!(link_args - pre_link_args_json, "pre-link-args");
        target_option_val!(link_args - late_link_args_json, "late-link-args");
        target_option_val!(link_args - late_link_args_dynamic_json, "late-link-args-dynamic");
        target_option_val!(link_args - late_link_args_static_json, "late-link-args-static");
        target_option_val!(link_args - post_link_args_json, "post-link-args");
        target_option_val!(link_script);
        target_option_val!(env - link_env);
        target_option_val!(link_env_remove);
        target_option_val!(asm_args);
        target_option_val!(cpu);
        target_option_val!(features);
        target_option_val!(dynamic_linking);
        target_option_val!(dll_tls_export);
        target_option_val!(only_cdylib);
        target_option_val!(executables);
        target_option_val!(relocation_model);
        target_option_val!(code_model);
        target_option_val!(tls_model);
        target_option_val!(disable_redzone);
        target_option_val!(frame_pointer);
        target_option_val!(function_sections);
        target_option_val!(dll_prefix);
        target_option_val!(dll_suffix);
        target_option_val!(exe_suffix);
        target_option_val!(staticlib_prefix);
        target_option_val!(staticlib_suffix);
        target_option_val!(families, "target-family");
        target_option_val!(abi_return_struct_as_int);
        target_option_val!(is_like_aix);
        target_option_val!(is_like_osx);
        target_option_val!(is_like_solaris);
        target_option_val!(is_like_windows);
        target_option_val!(is_like_msvc);
        target_option_val!(is_like_wasm);
        target_option_val!(is_like_android);
        target_option_val!(default_dwarf_version);
        target_option_val!(allows_weak_linkage);
        target_option_val!(has_rpath);
        target_option_val!(no_default_libraries);
        target_option_val!(position_independent_executables);
        target_option_val!(static_position_independent_executables);
        target_option_val!(plt_by_default);
        target_option_val!(relro_level);
        target_option_val!(archive_format);
        target_option_val!(allow_asm);
        target_option_val!(main_needs_argc_argv);
        target_option_val!(has_thread_local);
        target_option_val!(obj_is_bitcode);
        target_option_val!(forces_embed_bitcode);
        target_option_val!(bitcode_llvm_cmdline);
        target_option_val!(min_atomic_width);
        target_option_val!(max_atomic_width);
        target_option_val!(atomic_cas);
        target_option_val!(panic_strategy);
        target_option_val!(crt_static_allows_dylibs);
        target_option_val!(crt_static_default);
        target_option_val!(crt_static_respected);
        target_option_val!(stack_probes);
        target_option_val!(min_global_align);
        target_option_val!(default_codegen_units);
        target_option_val!(trap_unreachable);
        target_option_val!(requires_lto);
        target_option_val!(singlethread);
        target_option_val!(no_builtins);
        target_option_val!(default_hidden_visibility);
        target_option_val!(emit_debug_gdb_scripts);
        target_option_val!(requires_uwtable);
        target_option_val!(default_uwtable);
        target_option_val!(simd_types_indirect);
        target_option_val!(limit_rdylib_exports);
        target_option_val!(override_export_symbols);
        target_option_val!(merge_functions);
        target_option_val!(mcount, "target-mcount");
        target_option_val!(llvm_abiname);
        target_option_val!(relax_elf_relocations);
        target_option_val!(llvm_args);
        target_option_val!(use_ctors_section);
        target_option_val!(eh_frame_header);
        target_option_val!(has_thumb_interworking);
        target_option_val!(debuginfo_kind);
        target_option_val!(split_debuginfo);
        target_option_val!(supported_split_debuginfo);
        target_option_val!(supported_sanitizers);
        target_option_val!(c_enum_min_bits);
        target_option_val!(generate_arange_section);
        target_option_val!(supports_stack_protector);
        target_option_val!(entry_name);
        target_option_val!(entry_abi);
        target_option_val!(supports_xray);
        target_option_val!(force_emulated_tls);

        if let Some(abi) = self.default_adjusted_cabi {
            d.insert("default-adjusted-cabi".into(), Abi::name(abi).to_json());
        }

        Json::Object(d)
    }
}

/// Either a target triple string or a path to a JSON file.
#[derive(Clone, Debug)]
pub enum TargetTriple {
    TargetTriple(String),
    TargetJson {
        /// Warning: This field may only be used by rustdoc. Using it anywhere else will lead to
        /// inconsistencies as it is discarded during serialization.
        path_for_rustdoc: PathBuf,
        triple: String,
        contents: String,
    },
}

// Use a manual implementation to ignore the path field
impl PartialEq for TargetTriple {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::TargetTriple(l0), Self::TargetTriple(r0)) => l0 == r0,
            (
                Self::TargetJson { path_for_rustdoc: _, triple: l_triple, contents: l_contents },
                Self::TargetJson { path_for_rustdoc: _, triple: r_triple, contents: r_contents },
            ) => l_triple == r_triple && l_contents == r_contents,
            _ => false,
        }
    }
}

// Use a manual implementation to ignore the path field
impl Hash for TargetTriple {
    fn hash<H: Hasher>(&self, state: &mut H) -> () {
        match self {
            TargetTriple::TargetTriple(triple) => {
                0u8.hash(state);
                triple.hash(state)
            }
            TargetTriple::TargetJson { path_for_rustdoc: _, triple, contents } => {
                1u8.hash(state);
                triple.hash(state);
                contents.hash(state)
            }
        }
    }
}

// Use a manual implementation to prevent encoding the target json file path in the crate metadata
impl<S: Encoder> Encodable<S> for TargetTriple {
    fn encode(&self, s: &mut S) {
        match self {
            TargetTriple::TargetTriple(triple) => s.emit_enum_variant(0, |s| s.emit_str(triple)),
            TargetTriple::TargetJson { path_for_rustdoc: _, triple, contents } => s
                .emit_enum_variant(1, |s| {
                    s.emit_str(triple);
                    s.emit_str(contents)
                }),
        }
    }
}

impl<D: Decoder> Decodable<D> for TargetTriple {
    fn decode(d: &mut D) -> Self {
        match d.read_usize() {
            0 => TargetTriple::TargetTriple(d.read_str().to_owned()),
            1 => TargetTriple::TargetJson {
                path_for_rustdoc: PathBuf::new(),
                triple: d.read_str().to_owned(),
                contents: d.read_str().to_owned(),
            },
            _ => {
                panic!("invalid enum variant tag while decoding `TargetTriple`, expected 0..2");
            }
        }
    }
}

impl TargetTriple {
    /// Creates a target triple from the passed target triple string.
    pub fn from_triple(triple: &str) -> Self {
        TargetTriple::TargetTriple(triple.into())
    }

    /// Creates a target triple from the passed target path.
    pub fn from_path(path: &Path) -> Result<Self, io::Error> {
        let canonicalized_path = try_canonicalize(path)?;
        let contents = std::fs::read_to_string(&canonicalized_path).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("target path {canonicalized_path:?} is not a valid file: {err}"),
            )
        })?;
        let triple = canonicalized_path
            .file_stem()
            .expect("target path must not be empty")
            .to_str()
            .expect("target path must be valid unicode")
            .to_owned();
        Ok(TargetTriple::TargetJson { path_for_rustdoc: canonicalized_path, triple, contents })
    }

    /// Returns a string triple for this target.
    ///
    /// If this target is a path, the file name (without extension) is returned.
    pub fn triple(&self) -> &str {
        match *self {
            TargetTriple::TargetTriple(ref triple)
            | TargetTriple::TargetJson { ref triple, .. } => triple,
        }
    }

    /// Returns an extended string triple for this target.
    ///
    /// If this target is a path, a hash of the path is appended to the triple returned
    /// by `triple()`.
    pub fn debug_triple(&self) -> String {
        use std::collections::hash_map::DefaultHasher;

        match self {
            TargetTriple::TargetTriple(triple) => triple.to_owned(),
            TargetTriple::TargetJson { path_for_rustdoc: _, triple, contents: content } => {
                let mut hasher = DefaultHasher::new();
                content.hash(&mut hasher);
                let hash = hasher.finish();
                format!("{triple}-{hash}")
            }
        }
    }
}

impl fmt::Display for TargetTriple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.debug_triple())
    }
}
