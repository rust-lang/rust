//! [Flexible target specification.](https://github.com/rust-lang/rfcs/pull/131)
//!
//! Rust targets a wide variety of usecases, and in the interest of flexibility,
//! allows new target tuples to be defined in configuration files. Most users
//! will not need to care about these, but this is invaluable when porting Rust
//! to a new platform, and allows for an unprecedented level of control over how
//! the compiler works.
//!
//! # Using targets and target.json
//!
//! Invoking "rustc --target=${TUPLE}" will result in rustc initiating the [`Target::search`] by
//! - checking if "$TUPLE" is a complete path to a json (ending with ".json") and loading if so
//! - checking builtin targets for "${TUPLE}"
//! - checking directories in "${RUST_TARGET_PATH}" for "${TUPLE}.json"
//! - checking for "${RUSTC_SYSROOT}/lib/rustlib/${TUPLE}/target.json"
//!
//! Code will then be compiled using the first discovered target spec.
//!
//! # Defining a new target
//!
//! Targets are defined using a struct which additionally has serialization to and from [JSON].
//! The `Target` struct in this module loosely corresponds with the format the JSON takes.
//! We usually try to make the fields equivalent but we have given up on a 1:1 correspondence
//! between the JSON and the actual structure itself.
//!
//! Some fields are required in every target spec, and they should be embedded in Target directly.
//! Optional keys are in TargetOptions, but Target derefs to it, for no practical difference.
//! Most notable is the "data-layout" field which specifies Rust's notion of sizes and alignments
//! for several key types, such as f64, pointers, and so on.
//!
//! At one point we felt `-C` options should override the target's settings, like in C compilers,
//! but that was an essentially-unmarked route for making code incorrect and Rust unsound.
//! Confronted with programmers who prefer a compiler with a good UX instead of a lethal weapon,
//! we have almost-entirely recanted that notion, though we hope "target modifiers" will offer
//! a way to have a decent UX yet still extend the necessary compiler controls, without
//! requiring a new target spec for each and every single possible target micro-variant.
//!
//! [JSON]: https://json.org

use core::result::Result;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{fmt, io};

use rustc_abi::{
    Align, CanonAbi, Endian, ExternAbi, Integer, Size, TargetDataLayout, TargetDataLayoutErrors,
};
use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_error_messages::{DiagArgValue, IntoDiagArg, into_diag_arg_using_display};
use rustc_fs_util::try_canonicalize;
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_span::{Symbol, kw, sym};
use serde_json::Value;
use tracing::debug;

use crate::json::{Json, ToJson};
use crate::spec::crt_objects::CrtObjects;

pub mod crt_objects;

mod abi_map;
mod base;
mod json;

pub use abi_map::{AbiMap, AbiMapping};
pub use base::apple;
pub use base::avr::ef_avr_arch;
pub use json::json_schema;

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
/// and target properties like `is_like_windows`/`is_like_darwin`/etc. However, the PRs originally
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
    /// LLVM bitcode linker that can be used as a `self-contained` linker
    Llbc,
}

/// Linker flavors available externally through command line (`-Clinker-flavor`)
/// or json target specifications.
/// This set has accumulated historically, and contains both (stable and unstable) legacy values, as
/// well as modern ones matching the internal linker flavors (`LinkerFlavor`).
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LinkerFlavorCli {
    // Modern (unstable) flavors, with direct counterparts in `LinkerFlavor`.
    Gnu(Cc, Lld),
    Darwin(Cc, Lld),
    WasmLld(Cc),
    Unix(Cc),
    // Note: `Msvc(Lld::No)` is also a stable value.
    Msvc(Lld),
    EmCc,
    Bpf,
    Ptx,
    Llbc,

    // Legacy stable values
    Gcc,
    Ld,
    Lld(LldFlavor),
    Em,
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
            | LinkerFlavorCli::Llbc
            | LinkerFlavorCli::Ptx => true,
            LinkerFlavorCli::Gcc
            | LinkerFlavorCli::Ld
            | LinkerFlavorCli::Lld(..)
            | LinkerFlavorCli::Msvc(Lld::No)
            | LinkerFlavorCli::Em => false,
        }
    }
}

crate::target_spec_enum! {
    pub enum LldFlavor {
        Wasm = "wasm",
        Ld64 = "darwin",
        Ld = "gnu",
        Link = "link",
    }

    parse_error_type = "LLD flavor";
}

impl LinkerFlavor {
    /// At this point the target's reference linker flavor doesn't yet exist and we need to infer
    /// it. The inference always succeeds and gives some result, and we don't report any flavor
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
            LinkerFlavorCli::Llbc => LinkerFlavor::Llbc,
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
        }
    }

    /// Returns the corresponding backwards-compatible CLI flavor.
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
            LinkerFlavor::Bpf => LinkerFlavorCli::Bpf,
            LinkerFlavor::Llbc => LinkerFlavorCli::Llbc,
            LinkerFlavor::Ptx => LinkerFlavorCli::Ptx,
        }
    }

    /// Returns the modern CLI flavor that is the counterpart of this flavor.
    fn to_cli_counterpart(self) -> LinkerFlavorCli {
        match self {
            LinkerFlavor::Gnu(cc, lld) => LinkerFlavorCli::Gnu(cc, lld),
            LinkerFlavor::Darwin(cc, lld) => LinkerFlavorCli::Darwin(cc, lld),
            LinkerFlavor::WasmLld(cc) => LinkerFlavorCli::WasmLld(cc),
            LinkerFlavor::Unix(cc) => LinkerFlavorCli::Unix(cc),
            LinkerFlavor::Msvc(lld) => LinkerFlavorCli::Msvc(lld),
            LinkerFlavor::EmCc => LinkerFlavorCli::EmCc,
            LinkerFlavor::Bpf => LinkerFlavorCli::Bpf,
            LinkerFlavor::Llbc => LinkerFlavorCli::Llbc,
            LinkerFlavor::Ptx => LinkerFlavorCli::Ptx,
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
            LinkerFlavorCli::Llbc => (None, None),

            // Below: legacy stable values
            LinkerFlavorCli::Gcc => (Some(Cc::Yes), None),
            LinkerFlavorCli::Ld => (Some(Cc::No), Some(Lld::No)),
            LinkerFlavorCli::Lld(_) => (Some(Cc::No), Some(Lld::Yes)),
            LinkerFlavorCli::Em => (Some(Cc::Yes), Some(Lld::Yes)),
        }
    }

    fn infer_linker_hints(linker_stem: &str) -> Result<Self, (Option<Cc>, Option<Lld>)> {
        // Remove any version postfix.
        let stem = linker_stem
            .rsplit_once('-')
            .and_then(|(lhs, rhs)| rhs.chars().all(char::is_numeric).then_some(lhs))
            .unwrap_or(linker_stem);

        if stem == "llvm-bitcode-linker" {
            Ok(Self::Llbc)
        } else if stem == "emcc" // GCC/Clang can have an optional target prefix.
            || stem == "gcc"
            || stem.ends_with("-gcc")
            || stem == "g++"
            || stem.ends_with("-g++")
            || stem == "clang"
            || stem.ends_with("-clang")
            || stem == "clang++"
            || stem.ends_with("-clang++")
        {
            Err((Some(Cc::Yes), Some(Lld::No)))
        } else if stem == "wasm-ld"
            || stem.ends_with("-wasm-ld")
            || stem == "ld.lld"
            || stem == "lld"
            || stem == "rust-lld"
            || stem == "lld-link"
        {
            Err((Some(Cc::No), Some(Lld::Yes)))
        } else if stem == "ld" || stem.ends_with("-ld") || stem == "link" {
            Err((Some(Cc::No), Some(Lld::No)))
        } else {
            Err((None, None))
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
            LinkerFlavor::EmCc | LinkerFlavor::Bpf | LinkerFlavor::Llbc | LinkerFlavor::Ptx => self,
        }
    }

    pub fn with_cli_hints(self, cli: LinkerFlavorCli) -> LinkerFlavor {
        self.with_hints(LinkerFlavor::infer_cli_hints(cli))
    }

    pub fn with_linker_hints(self, linker_stem: &str) -> LinkerFlavor {
        match LinkerFlavor::infer_linker_hints(linker_stem) {
            Ok(linker_flavor) => linker_flavor,
            Err(hints) => self.with_hints(hints),
        }
    }

    pub fn check_compatibility(self, cli: LinkerFlavorCli) -> Option<String> {
        let compatible = |cli| {
            // The CLI flavor should be compatible with the target if:
            match (self, cli) {
                // 1. they are counterparts: they have the same principal flavor.
                (LinkerFlavor::Gnu(..), LinkerFlavorCli::Gnu(..))
                | (LinkerFlavor::Darwin(..), LinkerFlavorCli::Darwin(..))
                | (LinkerFlavor::WasmLld(..), LinkerFlavorCli::WasmLld(..))
                | (LinkerFlavor::Unix(..), LinkerFlavorCli::Unix(..))
                | (LinkerFlavor::Msvc(..), LinkerFlavorCli::Msvc(..))
                | (LinkerFlavor::EmCc, LinkerFlavorCli::EmCc)
                | (LinkerFlavor::Bpf, LinkerFlavorCli::Bpf)
                | (LinkerFlavor::Llbc, LinkerFlavorCli::Llbc)
                | (LinkerFlavor::Ptx, LinkerFlavorCli::Ptx) => return true,
                // 2. The linker flavor is independent of target and compatible
                (LinkerFlavor::Ptx, LinkerFlavorCli::Llbc) => return true,
                _ => {}
            }

            // 3. or, the flavor is legacy and survives this roundtrip.
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
            | LinkerFlavor::Llbc
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
            | LinkerFlavor::Llbc
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
            | LinkerFlavor::Llbc
            | LinkerFlavor::Ptx => false,
        }
    }

    /// For flavors with an `Lld` component, ensure it's enabled. Otherwise, returns the given
    /// flavor unmodified.
    pub fn with_lld_enabled(self) -> LinkerFlavor {
        match self {
            LinkerFlavor::Gnu(cc, Lld::No) => LinkerFlavor::Gnu(cc, Lld::Yes),
            LinkerFlavor::Darwin(cc, Lld::No) => LinkerFlavor::Darwin(cc, Lld::Yes),
            LinkerFlavor::Msvc(Lld::No) => LinkerFlavor::Msvc(Lld::Yes),
            _ => self,
        }
    }

    /// For flavors with an `Lld` component, ensure it's disabled. Otherwise, returns the given
    /// flavor unmodified.
    pub fn with_lld_disabled(self) -> LinkerFlavor {
        match self {
            LinkerFlavor::Gnu(cc, Lld::Yes) => LinkerFlavor::Gnu(cc, Lld::No),
            LinkerFlavor::Darwin(cc, Lld::Yes) => LinkerFlavor::Darwin(cc, Lld::No),
            LinkerFlavor::Msvc(Lld::Yes) => LinkerFlavor::Msvc(Lld::No),
            _ => self,
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

            pub fn desc(self) -> &'static str {
                match self {
                    $($($flavor)* => $string,)*
                }
            }
        }

        impl FromStr for LinkerFlavorCli {
            type Err = String;

            fn from_str(s: &str) -> Result<LinkerFlavorCli, Self::Err> {
                Ok(match s {
                    $($string => $($flavor)*,)*
                    _ => return Err(format!("invalid linker flavor, allowed values: {}", Self::one_of())),
                })
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
    (LinkerFlavorCli::Llbc) "llbc"
    (LinkerFlavorCli::Ptx) "ptx"

    // Legacy stable flavors
    (LinkerFlavorCli::Gcc) "gcc"
    (LinkerFlavorCli::Ld) "ld"
    (LinkerFlavorCli::Lld(LldFlavor::Ld)) "ld.lld"
    (LinkerFlavorCli::Lld(LldFlavor::Ld64)) "ld64.lld"
    (LinkerFlavorCli::Lld(LldFlavor::Link)) "lld-link"
    (LinkerFlavorCli::Lld(LldFlavor::Wasm)) "wasm-ld"
    (LinkerFlavorCli::Em) "em"
}

crate::json::serde_deserialize_from_str!(LinkerFlavorCli);
impl schemars::JsonSchema for LinkerFlavorCli {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "LinkerFlavor".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let all: Vec<&'static str> =
            Self::all().iter().map(|flavor| flavor.desc()).collect::<Vec<_>>();
        schemars::json_schema! ({
            "type": "string",
            "enum": all
        })
        .into()
    }
}

impl ToJson for LinkerFlavorCli {
    fn to_json(&self) -> Json {
        self.desc().to_json()
    }
}

/// The different `-Clink-self-contained` options that can be specified in a target spec:
/// - enabling or disabling in bulk
/// - some target-specific pieces of inference to determine whether to use self-contained linking
///   if `-Clink-self-contained` is not specified explicitly (e.g. on musl/mingw)
/// - explicitly enabling some of the self-contained linking components, e.g. the linker component
///   to use `rust-lld`
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LinkSelfContainedDefault {
    /// The target spec explicitly enables self-contained linking.
    True,

    /// The target spec explicitly disables self-contained linking.
    False,

    /// The target spec requests that the self-contained mode is inferred, in the context of musl.
    InferredForMusl,

    /// The target spec requests that the self-contained mode is inferred, in the context of mingw.
    InferredForMingw,

    /// The target spec explicitly enables a list of self-contained linking components: e.g. for
    /// targets opting into a subset of components like the CLI's `-C link-self-contained=+linker`.
    WithComponents(LinkSelfContainedComponents),
}

/// Parses a backwards-compatible `-Clink-self-contained` option string, without components.
impl FromStr for LinkSelfContainedDefault {
    type Err = String;

    fn from_str(s: &str) -> Result<LinkSelfContainedDefault, Self::Err> {
        Ok(match s {
            "false" => LinkSelfContainedDefault::False,
            "true" | "wasm" => LinkSelfContainedDefault::True,
            "musl" => LinkSelfContainedDefault::InferredForMusl,
            "mingw" => LinkSelfContainedDefault::InferredForMingw,
            _ => {
                return Err(format!(
                    "'{s}' is not a valid `-Clink-self-contained` default. \
                        Use 'false', 'true', 'wasm', 'musl' or 'mingw'",
                ));
            }
        })
    }
}

crate::json::serde_deserialize_from_str!(LinkSelfContainedDefault);
impl schemars::JsonSchema for LinkSelfContainedDefault {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "LinkSelfContainedDefault".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema! ({
            "type": "string",
            "enum": ["false", "true", "wasm", "musl", "mingw"]
        })
        .into()
    }
}

impl ToJson for LinkSelfContainedDefault {
    fn to_json(&self) -> Json {
        match *self {
            LinkSelfContainedDefault::WithComponents(components) => {
                // Serialize the components in a json object's `components` field, to prepare for a
                // future where `crt-objects-fallback` is removed from the json specs and
                // incorporated as a field here.
                let mut map = BTreeMap::new();
                map.insert("components", components);
                map.to_json()
            }

            // Stable backwards-compatible values
            LinkSelfContainedDefault::True => "true".to_json(),
            LinkSelfContainedDefault::False => "false".to_json(),
            LinkSelfContainedDefault::InferredForMusl => "musl".to_json(),
            LinkSelfContainedDefault::InferredForMingw => "mingw".to_json(),
        }
    }
}

impl LinkSelfContainedDefault {
    /// Returns whether the target spec has self-contained linking explicitly disabled. Used to emit
    /// errors if the user then enables it on the CLI.
    pub fn is_disabled(self) -> bool {
        self == LinkSelfContainedDefault::False
    }

    /// Returns the key to use when serializing the setting to json:
    /// - individual components in a `link-self-contained` object value
    /// - the other variants as a backwards-compatible `crt-objects-fallback` string
    fn json_key(self) -> &'static str {
        match self {
            LinkSelfContainedDefault::WithComponents(_) => "link-self-contained",
            _ => "crt-objects-fallback",
        }
    }

    /// Creates a `LinkSelfContainedDefault` enabling the self-contained linker for target specs
    /// (the equivalent of `-Clink-self-contained=+linker` on the CLI).
    pub fn with_linker() -> LinkSelfContainedDefault {
        LinkSelfContainedDefault::WithComponents(LinkSelfContainedComponents::LINKER)
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Default)]
    /// The `-C link-self-contained` components that can individually be enabled or disabled.
    pub struct LinkSelfContainedComponents: u8 {
        /// CRT objects (e.g. on `windows-gnu`, `musl`, `wasi` targets)
        const CRT_OBJECTS = 1 << 0;
        /// libc static library (e.g. on `musl`, `wasi` targets)
        const LIBC        = 1 << 1;
        /// libgcc/libunwind (e.g. on `windows-gnu`, `fuchsia`, `fortanix`, `gnullvm` targets)
        const UNWIND      = 1 << 2;
        /// Linker, dlltool, and their necessary libraries (e.g. on `windows-gnu` and for `rust-lld`)
        const LINKER      = 1 << 3;
        /// Sanitizer runtime libraries
        const SANITIZERS  = 1 << 4;
        /// Other MinGW libs and Windows import libs
        const MINGW       = 1 << 5;
    }
}
rustc_data_structures::external_bitflags_debug! { LinkSelfContainedComponents }

impl LinkSelfContainedComponents {
    /// Return the component's name.
    ///
    /// Returns `None` if the bitflags aren't a singular component (but a mix of multiple flags).
    pub fn as_str(self) -> Option<&'static str> {
        Some(match self {
            LinkSelfContainedComponents::CRT_OBJECTS => "crto",
            LinkSelfContainedComponents::LIBC => "libc",
            LinkSelfContainedComponents::UNWIND => "unwind",
            LinkSelfContainedComponents::LINKER => "linker",
            LinkSelfContainedComponents::SANITIZERS => "sanitizers",
            LinkSelfContainedComponents::MINGW => "mingw",
            _ => return None,
        })
    }

    /// Returns an array of all the components.
    fn all_components() -> [LinkSelfContainedComponents; 6] {
        [
            LinkSelfContainedComponents::CRT_OBJECTS,
            LinkSelfContainedComponents::LIBC,
            LinkSelfContainedComponents::UNWIND,
            LinkSelfContainedComponents::LINKER,
            LinkSelfContainedComponents::SANITIZERS,
            LinkSelfContainedComponents::MINGW,
        ]
    }

    /// Returns whether at least a component is enabled.
    pub fn are_any_components_enabled(self) -> bool {
        !self.is_empty()
    }

    /// Returns whether `LinkSelfContainedComponents::LINKER` is enabled.
    pub fn is_linker_enabled(self) -> bool {
        self.contains(LinkSelfContainedComponents::LINKER)
    }

    /// Returns whether `LinkSelfContainedComponents::CRT_OBJECTS` is enabled.
    pub fn is_crt_objects_enabled(self) -> bool {
        self.contains(LinkSelfContainedComponents::CRT_OBJECTS)
    }
}

impl FromStr for LinkSelfContainedComponents {
    type Err = String;

    /// Parses a single `-Clink-self-contained` well-known component, not a set of flags.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "crto" => LinkSelfContainedComponents::CRT_OBJECTS,
            "libc" => LinkSelfContainedComponents::LIBC,
            "unwind" => LinkSelfContainedComponents::UNWIND,
            "linker" => LinkSelfContainedComponents::LINKER,
            "sanitizers" => LinkSelfContainedComponents::SANITIZERS,
            "mingw" => LinkSelfContainedComponents::MINGW,
            _ => {
                return Err(format!(
                    "'{s}' is not a valid link-self-contained component, expected 'crto', 'libc', 'unwind', 'linker', 'sanitizers', 'mingw'"
                ));
            }
        })
    }
}

crate::json::serde_deserialize_from_str!(LinkSelfContainedComponents);
impl schemars::JsonSchema for LinkSelfContainedComponents {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "LinkSelfContainedComponents".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let all =
            Self::all_components().iter().map(|component| component.as_str()).collect::<Vec<_>>();
        schemars::json_schema! ({
            "type": "string",
            "enum": all,
        })
        .into()
    }
}

impl ToJson for LinkSelfContainedComponents {
    fn to_json(&self) -> Json {
        let components: Vec<_> = Self::all_components()
            .into_iter()
            .filter(|c| self.contains(*c))
            .map(|c| {
                // We can unwrap because we're iterating over all the known singular components,
                // not an actual set of flags where `as_str` can fail.
                c.as_str().unwrap().to_owned()
            })
            .collect();

        components.to_json()
    }
}

bitflags::bitflags! {
    /// The `-C linker-features` components that can individually be enabled or disabled.
    ///
    /// They are feature flags intended to be a more flexible mechanism than linker flavors, and
    /// also to prevent a combinatorial explosion of flavors whenever a new linker feature is
    /// required. These flags are "generic", in the sense that they can work on multiple targets on
    /// the CLI. Otherwise, one would have to select different linkers flavors for each target.
    ///
    /// Here are some examples of the advantages they offer:
    /// - default feature sets for principal flavors, or for specific targets.
    /// - flavor-specific features: for example, clang offers automatic cross-linking with
    ///   `--target`, which gcc-style compilers don't support. The *flavor* is still a C/C++
    ///   compiler, and we don't need to multiply the number of flavors for this use-case. Instead,
    ///   we can have a single `+target` feature.
    /// - umbrella features: for example if clang accumulates more features in the future than just
    ///   the `+target` above. That could be modeled as `+clang`.
    /// - niche features for resolving specific issues: for example, on Apple targets the linker
    ///   flag implementing the `as-needed` native link modifier (#99424) is only possible on
    ///   sufficiently recent linker versions.
    /// - still allows for discovery and automation, for example via feature detection. This can be
    ///   useful in exotic environments/build systems.
    #[derive(Clone, Copy, PartialEq, Eq, Default)]
    pub struct LinkerFeatures: u8 {
        /// Invoke the linker via a C/C++ compiler (e.g. on most unix targets).
        const CC  = 1 << 0;
        /// Use the lld linker, either the system lld or the self-contained linker `rust-lld`.
        const LLD = 1 << 1;
    }
}
rustc_data_structures::external_bitflags_debug! { LinkerFeatures }

impl LinkerFeatures {
    /// Parses a single `-C linker-features` well-known feature, not a set of flags.
    pub fn from_str(s: &str) -> Option<LinkerFeatures> {
        Some(match s {
            "cc" => LinkerFeatures::CC,
            "lld" => LinkerFeatures::LLD,
            _ => return None,
        })
    }

    /// Return the linker feature name, as would be passed on the CLI.
    ///
    /// Returns `None` if the bitflags aren't a singular component (but a mix of multiple flags).
    pub fn as_str(self) -> Option<&'static str> {
        Some(match self {
            LinkerFeatures::CC => "cc",
            LinkerFeatures::LLD => "lld",
            _ => return None,
        })
    }

    /// Returns whether the `lld` linker feature is enabled.
    pub fn is_lld_enabled(self) -> bool {
        self.contains(LinkerFeatures::LLD)
    }

    /// Returns whether the `cc` linker feature is enabled.
    pub fn is_cc_enabled(self) -> bool {
        self.contains(LinkerFeatures::CC)
    }
}

crate::target_spec_enum! {
    #[derive(Encodable, Decodable, HashStable_Generic)]
    pub enum PanicStrategy {
        Unwind = "unwind",
        Abort = "abort",
        ImmediateAbort = "immediate-abort",
    }

    parse_error_type = "panic strategy";
}

#[derive(Clone, Copy, Debug, PartialEq, Hash, Encodable, Decodable, HashStable_Generic)]
pub enum OnBrokenPipe {
    Default,
    Kill,
    Error,
    Inherit,
}

impl PanicStrategy {
    pub const fn desc_symbol(&self) -> Symbol {
        match *self {
            PanicStrategy::Unwind => sym::unwind,
            PanicStrategy::Abort => sym::abort,
            PanicStrategy::ImmediateAbort => sym::immediate_abort,
        }
    }

    pub fn unwinds(self) -> bool {
        matches!(self, PanicStrategy::Unwind)
    }
}

crate::target_spec_enum! {
    pub enum RelroLevel {
        Full = "full",
        Partial = "partial",
        Off = "off",
        None = "none",
    }

    parse_error_type = "relro level";
}

impl IntoDiagArg for PanicStrategy {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.desc().to_string()))
    }
}

crate::target_spec_enum! {
    pub enum SymbolVisibility {
        Hidden = "hidden",
        Protected = "protected",
        Interposable = "interposable",
    }

    parse_error_type = "symbol visibility";
}

#[derive(Clone, Debug, PartialEq, Hash)]
pub enum SmallDataThresholdSupport {
    None,
    DefaultForArch,
    LlvmModuleFlag(StaticCow<str>),
    LlvmArg(StaticCow<str>),
}

impl FromStr for SmallDataThresholdSupport {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "none" {
            Ok(Self::None)
        } else if s == "default-for-arch" {
            Ok(Self::DefaultForArch)
        } else if let Some(flag) = s.strip_prefix("llvm-module-flag=") {
            Ok(Self::LlvmModuleFlag(flag.to_string().into()))
        } else if let Some(arg) = s.strip_prefix("llvm-arg=") {
            Ok(Self::LlvmArg(arg.to_string().into()))
        } else {
            Err(format!("'{s}' is not a valid value for small-data-threshold-support."))
        }
    }
}

crate::json::serde_deserialize_from_str!(SmallDataThresholdSupport);
impl schemars::JsonSchema for SmallDataThresholdSupport {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SmallDataThresholdSupport".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema! ({
            "type": "string",
            "pattern": r#"^none|default-for-arch|llvm-module-flag=.+|llvm-arg=.+$"#,
        })
        .into()
    }
}

impl ToJson for SmallDataThresholdSupport {
    fn to_json(&self) -> Value {
        match self {
            Self::None => "none".to_json(),
            Self::DefaultForArch => "default-for-arch".to_json(),
            Self::LlvmModuleFlag(flag) => format!("llvm-module-flag={flag}").to_json(),
            Self::LlvmArg(arg) => format!("llvm-arg={arg}").to_json(),
        }
    }
}

crate::target_spec_enum! {
    pub enum MergeFunctions {
        Disabled = "disabled",
        Trampolines = "trampolines",
        Aliases = "aliases",
    }

    parse_error_type = "value for merge-functions";
}

crate::target_spec_enum! {
    pub enum RelocModel {
        Static = "static",
        Pic = "pic",
        Pie = "pie",
        DynamicNoPic = "dynamic-no-pic",
        Ropi = "ropi",
        Rwpi = "rwpi",
        RopiRwpi = "ropi-rwpi",
    }

    parse_error_type = "relocation model";
}

impl RelocModel {
    pub const fn desc_symbol(&self) -> Symbol {
        match *self {
            RelocModel::Static => kw::Static,
            RelocModel::Pic => sym::pic,
            RelocModel::Pie => sym::pie,
            RelocModel::DynamicNoPic => sym::dynamic_no_pic,
            RelocModel::Ropi => sym::ropi,
            RelocModel::Rwpi => sym::rwpi,
            RelocModel::RopiRwpi => sym::ropi_rwpi,
        }
    }
}

crate::target_spec_enum! {
    pub enum CodeModel {
        Tiny = "tiny",
        Small = "small",
        Kernel = "kernel",
        Medium = "medium",
        Large = "large",
    }

    parse_error_type = "code model";
}

crate::target_spec_enum! {
    /// The float ABI setting to be configured in the LLVM target machine.
    pub enum FloatAbi {
        Soft = "soft",
        Hard = "hard",
    }

    parse_error_type = "float abi";
}

crate::target_spec_enum! {
    /// The Rustc-specific variant of the ABI used for this target.
    pub enum RustcAbi {
        /// On x86-32 only: make use of SSE and SSE2 for ABI purposes.
        X86Sse2 = "x86-sse2",
        /// On x86-32/64 only: do not use any FPU or SIMD registers for the ABI.
        X86Softfloat = "x86-softfloat",
    }

    parse_error_type = "rustc abi";
}

crate::target_spec_enum! {
    pub enum TlsModel {
        GeneralDynamic = "global-dynamic",
        LocalDynamic = "local-dynamic",
        InitialExec = "initial-exec",
        LocalExec = "local-exec",
        Emulated = "emulated",
    }

    parse_error_type = "TLS model";
}

crate::target_spec_enum! {
    /// Everything is flattened to a single enum to make the json encoding/decoding less annoying.
    pub enum LinkOutputKind {
        /// Dynamically linked non position-independent executable.
        DynamicNoPicExe = "dynamic-nopic-exe",
        /// Dynamically linked position-independent executable.
        DynamicPicExe = "dynamic-pic-exe",
        /// Statically linked non position-independent executable.
        StaticNoPicExe = "static-nopic-exe",
        /// Statically linked position-independent executable.
        StaticPicExe = "static-pic-exe",
        /// Regular dynamic library ("dynamically linked").
        DynamicDylib = "dynamic-dylib",
        /// Dynamic library with bundled libc ("statically linked").
        StaticDylib = "static-dylib",
        /// WASI module with a lifetime past the _initialize entry point
        WasiReactorExe = "wasi-reactor-exe",
    }

    parse_error_type = "CRT object kind";
}

impl LinkOutputKind {
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

pub type LinkArgs = BTreeMap<LinkerFlavor, Vec<StaticCow<str>>>;
pub type LinkArgsCli = BTreeMap<LinkerFlavorCli, Vec<StaticCow<str>>>;

crate::target_spec_enum! {
    /// Which kind of debuginfo does the target use?
    ///
    /// Useful in determining whether a target supports Split DWARF (a target with
    /// `DebuginfoKind::Dwarf` and supporting `SplitDebuginfo::Unpacked` for example).
    #[derive(Default)]
    pub enum DebuginfoKind {
        /// DWARF debuginfo (such as that used on `x86_64_unknown_linux_gnu`).
        #[default]
        Dwarf = "dwarf",
        /// DWARF debuginfo in dSYM files (such as on Apple platforms).
        DwarfDsym = "dwarf-dsym",
        /// Program database files (such as on Windows).
        Pdb = "pdb",
    }

    parse_error_type = "debuginfo kind";
}

crate::target_spec_enum! {
    #[derive(Default)]
    pub enum SplitDebuginfo {
        /// Split debug-information is disabled, meaning that on supported platforms
        /// you can find all debug information in the executable itself. This is
        /// only supported for ELF effectively.
        ///
        /// * Windows - not supported
        /// * macOS - don't run `dsymutil`
        /// * ELF - `.debug_*` sections
        #[default]
        Off = "off",

        /// Split debug-information can be found in a "packed" location separate
        /// from the final artifact. This is supported on all platforms.
        ///
        /// * Windows - `*.pdb`
        /// * macOS - `*.dSYM` (run `dsymutil`)
        /// * ELF - `*.dwp` (run `thorin`)
        Packed = "packed",

        /// Split debug-information can be found in individual object files on the
        /// filesystem. The main executable may point to the object files.
        ///
        /// * Windows - not supported
        /// * macOS - supported, scattered object files
        /// * ELF - supported, scattered `*.dwo` or `*.o` files (see `SplitDwarfKind`)
        Unpacked = "unpacked",
    }

    parse_error_type = "split debuginfo";
}

into_diag_arg_using_display!(SplitDebuginfo);

#[derive(Clone, Debug, PartialEq, Eq, serde_derive::Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind")]
#[serde(rename_all = "kebab-case")]
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
    InlineOrCall {
        #[serde(rename = "min-llvm-version-for-inline")]
        min_llvm_version_for_inline: (u32, u32, u32),
    },
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

#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Encodable, Decodable, HashStable_Generic)]
pub struct SanitizerSet(u16);
bitflags::bitflags! {
    impl SanitizerSet: u16 {
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
        const DATAFLOW = 1 << 11;
    }
}
rustc_data_structures::external_bitflags_debug! { SanitizerSet }

impl SanitizerSet {
    // Taken from LLVM's sanitizer compatibility logic:
    // https://github.com/llvm/llvm-project/blob/release/18.x/clang/lib/Driver/SanitizerArgs.cpp#L512
    const MUTUALLY_EXCLUSIVE: &'static [(SanitizerSet, SanitizerSet)] = &[
        (SanitizerSet::ADDRESS, SanitizerSet::MEMORY),
        (SanitizerSet::ADDRESS, SanitizerSet::THREAD),
        (SanitizerSet::ADDRESS, SanitizerSet::HWADDRESS),
        (SanitizerSet::ADDRESS, SanitizerSet::MEMTAG),
        (SanitizerSet::ADDRESS, SanitizerSet::KERNELADDRESS),
        (SanitizerSet::ADDRESS, SanitizerSet::SAFESTACK),
        (SanitizerSet::LEAK, SanitizerSet::MEMORY),
        (SanitizerSet::LEAK, SanitizerSet::THREAD),
        (SanitizerSet::LEAK, SanitizerSet::KERNELADDRESS),
        (SanitizerSet::LEAK, SanitizerSet::SAFESTACK),
        (SanitizerSet::MEMORY, SanitizerSet::THREAD),
        (SanitizerSet::MEMORY, SanitizerSet::HWADDRESS),
        (SanitizerSet::MEMORY, SanitizerSet::KERNELADDRESS),
        (SanitizerSet::MEMORY, SanitizerSet::SAFESTACK),
        (SanitizerSet::THREAD, SanitizerSet::HWADDRESS),
        (SanitizerSet::THREAD, SanitizerSet::KERNELADDRESS),
        (SanitizerSet::THREAD, SanitizerSet::SAFESTACK),
        (SanitizerSet::HWADDRESS, SanitizerSet::MEMTAG),
        (SanitizerSet::HWADDRESS, SanitizerSet::KERNELADDRESS),
        (SanitizerSet::HWADDRESS, SanitizerSet::SAFESTACK),
        (SanitizerSet::CFI, SanitizerSet::KCFI),
        (SanitizerSet::MEMTAG, SanitizerSet::KERNELADDRESS),
        (SanitizerSet::KERNELADDRESS, SanitizerSet::SAFESTACK),
    ];

    /// Return sanitizer's name
    ///
    /// Returns none if the flags is a set of sanitizers numbering not exactly one.
    pub fn as_str(self) -> Option<&'static str> {
        Some(match self {
            SanitizerSet::ADDRESS => "address",
            SanitizerSet::CFI => "cfi",
            SanitizerSet::DATAFLOW => "dataflow",
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

    pub fn mutually_exclusive(self) -> Option<(SanitizerSet, SanitizerSet)> {
        Self::MUTUALLY_EXCLUSIVE
            .into_iter()
            .find(|&(a, b)| self.contains(*a) && self.contains(*b))
            .copied()
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

impl FromStr for SanitizerSet {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "address" => SanitizerSet::ADDRESS,
            "cfi" => SanitizerSet::CFI,
            "dataflow" => SanitizerSet::DATAFLOW,
            "kcfi" => SanitizerSet::KCFI,
            "kernel-address" => SanitizerSet::KERNELADDRESS,
            "leak" => SanitizerSet::LEAK,
            "memory" => SanitizerSet::MEMORY,
            "memtag" => SanitizerSet::MEMTAG,
            "safestack" => SanitizerSet::SAFESTACK,
            "shadow-call-stack" => SanitizerSet::SHADOWCALLSTACK,
            "thread" => SanitizerSet::THREAD,
            "hwaddress" => SanitizerSet::HWADDRESS,
            s => return Err(format!("unknown sanitizer {s}")),
        })
    }
}

crate::json::serde_deserialize_from_str!(SanitizerSet);
impl schemars::JsonSchema for SanitizerSet {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SanitizerSet".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let all = Self::all().iter().map(|sanitizer| sanitizer.as_str()).collect::<Vec<_>>();
        schemars::json_schema! ({
            "type": "string",
            "enum": all,
        })
        .into()
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

crate::target_spec_enum! {
    pub enum FramePointer {
        /// Forces the machine code generator to always preserve the frame pointers.
        Always = "always",
        /// Forces the machine code generator to preserve the frame pointers except for the leaf
        /// functions (i.e. those that don't call other functions).
        NonLeaf = "non-leaf",
        /// Allows the machine code generator to omit the frame pointers.
        ///
        /// This option does not guarantee that the frame pointers will be omitted.
        MayOmit = "may-omit",
    }

    parse_error_type = "frame pointer";
}

impl FramePointer {
    /// It is intended that the "force frame pointer" transition is "one way"
    /// so this convenience assures such if used
    #[inline]
    pub fn ratchet(&mut self, rhs: FramePointer) -> FramePointer {
        *self = match (*self, rhs) {
            (FramePointer::Always, _) | (_, FramePointer::Always) => FramePointer::Always,
            (FramePointer::NonLeaf, _) | (_, FramePointer::NonLeaf) => FramePointer::NonLeaf,
            _ => FramePointer::MayOmit,
        };
        *self
    }
}

crate::target_spec_enum! {
    /// Controls use of stack canaries.
    pub enum StackProtector {
        /// Disable stack canary generation.
        None = "none",

        /// On LLVM, mark all generated LLVM functions with the `ssp` attribute (see
        /// llvm/docs/LangRef.rst). This triggers stack canary generation in
        /// functions which contain an array of a byte-sized type with more than
        /// eight elements.
        Basic = "basic",

        /// On LLVM, mark all generated LLVM functions with the `sspstrong`
        /// attribute (see llvm/docs/LangRef.rst). This triggers stack canary
        /// generation in functions which either contain an array, or which take
        /// the address of a local variable.
        Strong = "strong",

        /// Generate stack canaries in all functions.
        All = "all",
    }

    parse_error_type = "stack protector";
}

into_diag_arg_using_display!(StackProtector);

crate::target_spec_enum! {
    pub enum BinaryFormat {
        Coff = "coff",
        Elf = "elf",
        MachO = "mach-o",
        Wasm = "wasm",
        Xcoff = "xcoff",
    }

    parse_error_type = "binary format";
}

impl BinaryFormat {
    /// Returns [`object::BinaryFormat`] for given `BinaryFormat`
    pub fn to_object(&self) -> object::BinaryFormat {
        match self {
            Self::Coff => object::BinaryFormat::Coff,
            Self::Elf => object::BinaryFormat::Elf,
            Self::MachO => object::BinaryFormat::MachO,
            Self::Wasm => object::BinaryFormat::Wasm,
            Self::Xcoff => object::BinaryFormat::Xcoff,
        }
    }
}

impl ToJson for Align {
    fn to_json(&self) -> Json {
        self.bits().to_json()
    }
}

macro_rules! supported_targets {
    ( $(($tuple:literal, $module:ident),)+ ) => {
        mod targets {
            $(pub(crate) mod $module;)+
        }

        /// List of supported targets
        pub static TARGETS: &[&str] = &[$($tuple),+];

        fn load_builtin(target: &str) -> Option<Target> {
            let t = match target {
                $( $tuple => targets::$module::target(), )+
                _ => return None,
            };
            debug!("got builtin target: {:?}", t);
            Some(t)
        }

        fn load_all_builtins() -> impl Iterator<Item = Target> {
            [
                $( targets::$module::target, )+
            ]
            .into_iter()
            .map(|f| f())
        }

        #[cfg(test)]
        mod tests {
            // Cannot put this into a separate file without duplication, make an exception.
            $(
                #[test] // `#[test]`
                fn $module() {
                    crate::spec::targets::$module::target().test_target()
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
    ("loongarch64-unknown-linux-musl", loongarch64_unknown_linux_musl),
    ("m68k-unknown-linux-gnu", m68k_unknown_linux_gnu),
    ("m68k-unknown-none-elf", m68k_unknown_none_elf),
    ("csky-unknown-linux-gnuabiv2", csky_unknown_linux_gnuabiv2),
    ("csky-unknown-linux-gnuabiv2hf", csky_unknown_linux_gnuabiv2hf),
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
    ("powerpc-unknown-linux-muslspe", powerpc_unknown_linux_muslspe),
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
    ("aarch64_be-unknown-linux-musl", aarch64_be_unknown_linux_musl),
    ("x86_64-unknown-linux-musl", x86_64_unknown_linux_musl),
    ("i686-unknown-linux-musl", i686_unknown_linux_musl),
    ("i586-unknown-linux-musl", i586_unknown_linux_musl),
    ("mips-unknown-linux-musl", mips_unknown_linux_musl),
    ("mipsel-unknown-linux-musl", mipsel_unknown_linux_musl),
    ("mips64-unknown-linux-muslabi64", mips64_unknown_linux_muslabi64),
    ("mips64el-unknown-linux-muslabi64", mips64el_unknown_linux_muslabi64),
    ("hexagon-unknown-linux-musl", hexagon_unknown_linux_musl),
    ("hexagon-unknown-none-elf", hexagon_unknown_none_elf),

    ("mips-unknown-linux-uclibc", mips_unknown_linux_uclibc),
    ("mipsel-unknown-linux-uclibc", mipsel_unknown_linux_uclibc),

    ("i686-linux-android", i686_linux_android),
    ("x86_64-linux-android", x86_64_linux_android),
    ("arm-linux-androideabi", arm_linux_androideabi),
    ("armv7-linux-androideabi", armv7_linux_androideabi),
    ("thumbv7neon-linux-androideabi", thumbv7neon_linux_androideabi),
    ("aarch64-linux-android", aarch64_linux_android),
    ("riscv64-linux-android", riscv64_linux_android),

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
    ("i586-unknown-netbsd", i586_unknown_netbsd),
    ("i686-unknown-netbsd", i686_unknown_netbsd),
    ("mipsel-unknown-netbsd", mipsel_unknown_netbsd),
    ("powerpc-unknown-netbsd", powerpc_unknown_netbsd),
    ("riscv64gc-unknown-netbsd", riscv64gc_unknown_netbsd),
    ("sparc64-unknown-netbsd", sparc64_unknown_netbsd),
    ("x86_64-unknown-netbsd", x86_64_unknown_netbsd),

    ("i686-unknown-haiku", i686_unknown_haiku),
    ("x86_64-unknown-haiku", x86_64_unknown_haiku),

    ("i686-unknown-hurd-gnu", i686_unknown_hurd_gnu),
    ("x86_64-unknown-hurd-gnu", x86_64_unknown_hurd_gnu),

    ("aarch64-apple-darwin", aarch64_apple_darwin),
    ("arm64e-apple-darwin", arm64e_apple_darwin),
    ("x86_64-apple-darwin", x86_64_apple_darwin),
    ("x86_64h-apple-darwin", x86_64h_apple_darwin),
    ("i686-apple-darwin", i686_apple_darwin),

    ("aarch64-unknown-fuchsia", aarch64_unknown_fuchsia),
    ("riscv64gc-unknown-fuchsia", riscv64gc_unknown_fuchsia),
    ("x86_64-unknown-fuchsia", x86_64_unknown_fuchsia),

    ("avr-none", avr_none),

    ("x86_64-unknown-l4re-uclibc", x86_64_unknown_l4re_uclibc),

    ("aarch64-unknown-redox", aarch64_unknown_redox),
    ("i586-unknown-redox", i586_unknown_redox),
    ("x86_64-unknown-redox", x86_64_unknown_redox),

    ("x86_64-unknown-managarm-mlibc", x86_64_unknown_managarm_mlibc),
    ("aarch64-unknown-managarm-mlibc", aarch64_unknown_managarm_mlibc),
    ("riscv64gc-unknown-managarm-mlibc", riscv64gc_unknown_managarm_mlibc),

    ("i386-apple-ios", i386_apple_ios),
    ("x86_64-apple-ios", x86_64_apple_ios),
    ("aarch64-apple-ios", aarch64_apple_ios),
    ("arm64e-apple-ios", arm64e_apple_ios),
    ("armv7s-apple-ios", armv7s_apple_ios),
    ("x86_64-apple-ios-macabi", x86_64_apple_ios_macabi),
    ("aarch64-apple-ios-macabi", aarch64_apple_ios_macabi),
    ("aarch64-apple-ios-sim", aarch64_apple_ios_sim),

    ("aarch64-apple-tvos", aarch64_apple_tvos),
    ("aarch64-apple-tvos-sim", aarch64_apple_tvos_sim),
    ("arm64e-apple-tvos", arm64e_apple_tvos),
    ("x86_64-apple-tvos", x86_64_apple_tvos),

    ("armv7k-apple-watchos", armv7k_apple_watchos),
    ("arm64_32-apple-watchos", arm64_32_apple_watchos),
    ("x86_64-apple-watchos-sim", x86_64_apple_watchos_sim),
    ("aarch64-apple-watchos", aarch64_apple_watchos),
    ("aarch64-apple-watchos-sim", aarch64_apple_watchos_sim),

    ("aarch64-apple-visionos", aarch64_apple_visionos),
    ("aarch64-apple-visionos-sim", aarch64_apple_visionos_sim),

    ("armebv7r-none-eabi", armebv7r_none_eabi),
    ("armebv7r-none-eabihf", armebv7r_none_eabihf),
    ("armv7r-none-eabi", armv7r_none_eabi),
    ("armv7r-none-eabihf", armv7r_none_eabihf),
    ("armv8r-none-eabihf", armv8r_none_eabihf),

    ("armv7-rtems-eabihf", armv7_rtems_eabihf),

    ("x86_64-pc-solaris", x86_64_pc_solaris),
    ("sparcv9-sun-solaris", sparcv9_sun_solaris),

    ("x86_64-unknown-illumos", x86_64_unknown_illumos),
    ("aarch64-unknown-illumos", aarch64_unknown_illumos),

    ("x86_64-pc-windows-gnu", x86_64_pc_windows_gnu),
    ("x86_64-uwp-windows-gnu", x86_64_uwp_windows_gnu),
    ("x86_64-win7-windows-gnu", x86_64_win7_windows_gnu),
    ("i686-pc-windows-gnu", i686_pc_windows_gnu),
    ("i686-uwp-windows-gnu", i686_uwp_windows_gnu),
    ("i686-win7-windows-gnu", i686_win7_windows_gnu),

    ("aarch64-pc-windows-gnullvm", aarch64_pc_windows_gnullvm),
    ("i686-pc-windows-gnullvm", i686_pc_windows_gnullvm),
    ("x86_64-pc-windows-gnullvm", x86_64_pc_windows_gnullvm),

    ("aarch64-pc-windows-msvc", aarch64_pc_windows_msvc),
    ("aarch64-uwp-windows-msvc", aarch64_uwp_windows_msvc),
    ("arm64ec-pc-windows-msvc", arm64ec_pc_windows_msvc),
    ("x86_64-pc-windows-msvc", x86_64_pc_windows_msvc),
    ("x86_64-uwp-windows-msvc", x86_64_uwp_windows_msvc),
    ("x86_64-win7-windows-msvc", x86_64_win7_windows_msvc),
    ("i686-pc-windows-msvc", i686_pc_windows_msvc),
    ("i686-uwp-windows-msvc", i686_uwp_windows_msvc),
    ("i686-win7-windows-msvc", i686_win7_windows_msvc),
    ("thumbv7a-pc-windows-msvc", thumbv7a_pc_windows_msvc),
    ("thumbv7a-uwp-windows-msvc", thumbv7a_uwp_windows_msvc),

    ("wasm32-unknown-emscripten", wasm32_unknown_emscripten),
    ("wasm32-unknown-unknown", wasm32_unknown_unknown),
    ("wasm32v1-none", wasm32v1_none),
    ("wasm32-wasip1", wasm32_wasip1),
    ("wasm32-wasip2", wasm32_wasip2),
    ("wasm32-wasip1-threads", wasm32_wasip1_threads),
    ("wasm32-wali-linux-musl", wasm32_wali_linux_musl),
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
    ("armv7a-nuttx-eabi", armv7a_nuttx_eabi),
    ("armv7a-nuttx-eabihf", armv7a_nuttx_eabihf),
    ("armv7a-vex-v5", armv7a_vex_v5),

    ("msp430-none-elf", msp430_none_elf),

    ("aarch64_be-unknown-hermit", aarch64_be_unknown_hermit),
    ("aarch64-unknown-hermit", aarch64_unknown_hermit),
    ("riscv64gc-unknown-hermit", riscv64gc_unknown_hermit),
    ("x86_64-unknown-hermit", x86_64_unknown_hermit),
    ("x86_64-unknown-motor", x86_64_unknown_motor),

    ("x86_64-unikraft-linux-musl", x86_64_unikraft_linux_musl),

    ("armv7-unknown-trusty", armv7_unknown_trusty),
    ("aarch64-unknown-trusty", aarch64_unknown_trusty),
    ("x86_64-unknown-trusty", x86_64_unknown_trusty),

    ("riscv32i-unknown-none-elf", riscv32i_unknown_none_elf),
    ("riscv32im-risc0-zkvm-elf", riscv32im_risc0_zkvm_elf),
    ("riscv32im-unknown-none-elf", riscv32im_unknown_none_elf),
    ("riscv32ima-unknown-none-elf", riscv32ima_unknown_none_elf),
    ("riscv32imc-unknown-none-elf", riscv32imc_unknown_none_elf),
    ("riscv32imc-esp-espidf", riscv32imc_esp_espidf),
    ("riscv32imac-esp-espidf", riscv32imac_esp_espidf),
    ("riscv32imafc-esp-espidf", riscv32imafc_esp_espidf),

    ("riscv32e-unknown-none-elf", riscv32e_unknown_none_elf),
    ("riscv32em-unknown-none-elf", riscv32em_unknown_none_elf),
    ("riscv32emc-unknown-none-elf", riscv32emc_unknown_none_elf),

    ("riscv32imac-unknown-none-elf", riscv32imac_unknown_none_elf),
    ("riscv32imafc-unknown-none-elf", riscv32imafc_unknown_none_elf),
    ("riscv32imac-unknown-xous-elf", riscv32imac_unknown_xous_elf),
    ("riscv32gc-unknown-linux-gnu", riscv32gc_unknown_linux_gnu),
    ("riscv32gc-unknown-linux-musl", riscv32gc_unknown_linux_musl),
    ("riscv64imac-unknown-none-elf", riscv64imac_unknown_none_elf),
    ("riscv64gc-unknown-none-elf", riscv64gc_unknown_none_elf),
    ("riscv64gc-unknown-linux-gnu", riscv64gc_unknown_linux_gnu),
    ("riscv64gc-unknown-linux-musl", riscv64gc_unknown_linux_musl),
    ("riscv64a23-unknown-linux-gnu", riscv64a23_unknown_linux_gnu),

    ("sparc-unknown-none-elf", sparc_unknown_none_elf),

    ("loongarch32-unknown-none", loongarch32_unknown_none),
    ("loongarch32-unknown-none-softfloat", loongarch32_unknown_none_softfloat),
    ("loongarch64-unknown-none", loongarch64_unknown_none),
    ("loongarch64-unknown-none-softfloat", loongarch64_unknown_none_softfloat),

    ("aarch64-unknown-none", aarch64_unknown_none),
    ("aarch64-unknown-none-softfloat", aarch64_unknown_none_softfloat),
    ("aarch64_be-unknown-none-softfloat", aarch64_be_unknown_none_softfloat),
    ("aarch64-unknown-nuttx", aarch64_unknown_nuttx),

    ("x86_64-fortanix-unknown-sgx", x86_64_fortanix_unknown_sgx),

    ("x86_64-unknown-uefi", x86_64_unknown_uefi),
    ("i686-unknown-uefi", i686_unknown_uefi),
    ("aarch64-unknown-uefi", aarch64_unknown_uefi),

    ("nvptx64-nvidia-cuda", nvptx64_nvidia_cuda),

    ("amdgcn-amd-amdhsa", amdgcn_amd_amdhsa),

    ("xtensa-esp32-none-elf", xtensa_esp32_none_elf),
    ("xtensa-esp32-espidf", xtensa_esp32_espidf),
    ("xtensa-esp32s2-none-elf", xtensa_esp32s2_none_elf),
    ("xtensa-esp32s2-espidf", xtensa_esp32s2_espidf),
    ("xtensa-esp32s3-none-elf", xtensa_esp32s3_none_elf),
    ("xtensa-esp32s3-espidf", xtensa_esp32s3_espidf),

    ("i686-wrs-vxworks", i686_wrs_vxworks),
    ("x86_64-wrs-vxworks", x86_64_wrs_vxworks),
    ("armv7-wrs-vxworks-eabihf", armv7_wrs_vxworks_eabihf),
    ("aarch64-wrs-vxworks", aarch64_wrs_vxworks),
    ("powerpc-wrs-vxworks", powerpc_wrs_vxworks),
    ("powerpc-wrs-vxworks-spe", powerpc_wrs_vxworks_spe),
    ("powerpc64-wrs-vxworks", powerpc64_wrs_vxworks),
    ("riscv32-wrs-vxworks", riscv32_wrs_vxworks),
    ("riscv64-wrs-vxworks", riscv64_wrs_vxworks),

    ("aarch64-kmc-solid_asp3", aarch64_kmc_solid_asp3),
    ("armv7a-kmc-solid_asp3-eabi", armv7a_kmc_solid_asp3_eabi),
    ("armv7a-kmc-solid_asp3-eabihf", armv7a_kmc_solid_asp3_eabihf),

    ("mipsel-sony-psp", mipsel_sony_psp),
    ("mipsel-sony-psx", mipsel_sony_psx),
    ("mipsel-unknown-none", mipsel_unknown_none),
    ("mips-mti-none-elf", mips_mti_none_elf),
    ("mipsel-mti-none-elf", mipsel_mti_none_elf),
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

    ("aarch64-unknown-teeos", aarch64_unknown_teeos),

    ("mips64-openwrt-linux-musl", mips64_openwrt_linux_musl),

    ("aarch64-unknown-nto-qnx700", aarch64_unknown_nto_qnx700),
    ("aarch64-unknown-nto-qnx710", aarch64_unknown_nto_qnx710),
    ("aarch64-unknown-nto-qnx710_iosock", aarch64_unknown_nto_qnx710_iosock),
    ("aarch64-unknown-nto-qnx800", aarch64_unknown_nto_qnx800),
    ("x86_64-pc-nto-qnx710", x86_64_pc_nto_qnx710),
    ("x86_64-pc-nto-qnx710_iosock", x86_64_pc_nto_qnx710_iosock),
    ("x86_64-pc-nto-qnx800", x86_64_pc_nto_qnx800),
    ("i686-pc-nto-qnx700", i686_pc_nto_qnx700),

    ("aarch64-unknown-linux-ohos", aarch64_unknown_linux_ohos),
    ("armv7-unknown-linux-ohos", armv7_unknown_linux_ohos),
    ("loongarch64-unknown-linux-ohos", loongarch64_unknown_linux_ohos),
    ("x86_64-unknown-linux-ohos", x86_64_unknown_linux_ohos),

    ("x86_64-unknown-linux-none", x86_64_unknown_linux_none),

    ("thumbv6m-nuttx-eabi", thumbv6m_nuttx_eabi),
    ("thumbv7a-nuttx-eabi", thumbv7a_nuttx_eabi),
    ("thumbv7a-nuttx-eabihf", thumbv7a_nuttx_eabihf),
    ("thumbv7m-nuttx-eabi", thumbv7m_nuttx_eabi),
    ("thumbv7em-nuttx-eabi", thumbv7em_nuttx_eabi),
    ("thumbv7em-nuttx-eabihf", thumbv7em_nuttx_eabihf),
    ("thumbv8m.base-nuttx-eabi", thumbv8m_base_nuttx_eabi),
    ("thumbv8m.main-nuttx-eabi", thumbv8m_main_nuttx_eabi),
    ("thumbv8m.main-nuttx-eabihf", thumbv8m_main_nuttx_eabihf),
    ("riscv32imc-unknown-nuttx-elf", riscv32imc_unknown_nuttx_elf),
    ("riscv32imac-unknown-nuttx-elf", riscv32imac_unknown_nuttx_elf),
    ("riscv32imafc-unknown-nuttx-elf", riscv32imafc_unknown_nuttx_elf),
    ("riscv64imac-unknown-nuttx-elf", riscv64imac_unknown_nuttx_elf),
    ("riscv64gc-unknown-nuttx-elf", riscv64gc_unknown_nuttx_elf),
    ("x86_64-lynx-lynxos178", x86_64_lynx_lynxos178),

    ("x86_64-pc-cygwin", x86_64_pc_cygwin),
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
}

impl TargetWarnings {
    pub fn empty() -> Self {
        Self { unused_fields: Vec::new() }
    }

    pub fn warning_messages(&self) -> Vec<String> {
        let mut warnings = vec![];
        if !self.unused_fields.is_empty() {
            warnings.push(format!(
                "target json file contains unused fields: {}",
                self.unused_fields.join(", ")
            ));
        }
        warnings
    }
}

/// For the [`Target::check_consistency`] function, determines whether the given target is a builtin or a JSON
/// target.
#[derive(Copy, Clone, Debug, PartialEq)]
enum TargetKind {
    Json,
    Builtin,
}

/// Everything `rustc` knows about how to compile for a specific target.
///
/// Every field here must be specified, and has no default value.
#[derive(PartialEq, Clone, Debug)]
pub struct Target {
    /// Unversioned target tuple to pass to LLVM.
    ///
    /// Target tuples can optionally contain an OS version (notably Apple targets), which rustc
    /// cannot know without querying the environment.
    ///
    /// Use `rustc_codegen_ssa::back::versioned_llvm_target` if you need the full LLVM target.
    pub llvm_target: StaticCow<str>,
    /// Metadata about a target, for example the description or tier.
    /// Used for generating target documentation.
    pub metadata: TargetMetadata,
    /// Number of bits in a pointer. Influences the `target_pointer_width` `cfg` variable.
    pub pointer_width: u16,
    /// Architecture to use for ABI considerations. Valid options include: "x86",
    /// "x86_64", "arm", "aarch64", "mips", "powerpc", "powerpc64", and others.
    pub arch: StaticCow<str>,
    /// [Data layout](https://llvm.org/docs/LangRef.html#data-layout) to pass to LLVM.
    pub data_layout: StaticCow<str>,
    /// Optional settings with defaults.
    pub options: TargetOptions,
}

/// Metadata about a target like the description or tier.
/// Part of #120745.
/// All fields are optional for now, but intended to be required in the future.
#[derive(Default, PartialEq, Clone, Debug)]
pub struct TargetMetadata {
    /// A short description of the target including platform requirements,
    /// for example "64-bit Linux (kernel 3.2+, glibc 2.17+)".
    pub description: Option<StaticCow<str>>,
    /// The tier of the target. 1, 2 or 3.
    pub tier: Option<u64>,
    /// Whether the Rust project ships host tools for a target.
    pub host_tools: Option<bool>,
    /// Whether a target has the `std` library. This is usually true for targets running
    /// on an operating system.
    pub std: Option<bool>,
}

impl Target {
    pub fn parse_data_layout(&self) -> Result<TargetDataLayout, TargetDataLayoutErrors<'_>> {
        let mut dl = TargetDataLayout::parse_from_llvm_datalayout_string(
            &self.data_layout,
            self.options.default_address_space,
        )?;

        // Perform consistency checks against the Target information.
        if dl.endian != self.endian {
            return Err(TargetDataLayoutErrors::InconsistentTargetArchitecture {
                dl: dl.endian.as_str(),
                target: self.endian.as_str(),
            });
        }

        let target_pointer_width: u64 = self.pointer_width.into();
        let dl_pointer_size: u64 = dl.pointer_size().bits();
        if dl_pointer_size != target_pointer_width {
            return Err(TargetDataLayoutErrors::InconsistentTargetPointerWidth {
                pointer_size: dl_pointer_size,
                target: self.pointer_width,
            });
        }

        dl.c_enum_min_size = Integer::from_size(Size::from_bits(
            self.c_enum_min_bits.unwrap_or(self.c_int_width as _),
        ))
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

/// x86 (32-bit) abi options.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct X86Abi {
    /// On x86-32 targets, the regparm N causes the compiler to pass arguments
    /// in registers EAX, EDX, and ECX instead of on the stack.
    pub regparm: Option<u32>,
    /// Override the default ABI to return small structs in registers
    pub reg_struct_return: bool,
}

pub trait HasX86AbiOpt {
    fn x86_abi_opt(&self) -> X86Abi;
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
    /// Used as the `target_endian` `cfg` variable. Defaults to little endian.
    pub endian: Endian,
    /// Width of c_int type. Defaults to "32".
    pub c_int_width: u16,
    /// OS name to use for conditional compilation (`target_os`). Defaults to "none".
    /// "none" implies a bare metal target without `std` library.
    /// A couple of targets having `std` also use "unknown" as an `os` value,
    /// but they are exceptions.
    pub os: StaticCow<str>,
    /// Environment name to use for conditional compilation (`target_env`). Defaults to "".
    pub env: StaticCow<str>,
    /// ABI name to distinguish multiple ABIs on the same OS and architecture. For instance, `"eabi"`
    /// or `"eabihf"`. Defaults to "".
    /// This field is *not* forwarded directly to LLVM; its primary purpose is `cfg(target_abi)`.
    /// However, parts of the backend do check this field for specific values to enable special behavior.
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
    /// Behavior for the self-contained linking mode: inferred for some targets, or explicitly
    /// enabled (in bulk, or with individual components).
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
    /// Whether a cpu needs to be explicitly set.
    /// Set to true if there is no default cpu. Defaults to false.
    pub need_explicit_cpu: bool,
    /// Default target features to pass to LLVM. These features overwrite
    /// `-Ctarget-cpu` but can be overwritten with `-Ctarget-features`.
    /// Corresponds to `llc -mattr=$features`.
    /// Note that these are LLVM feature names, not Rust feature names!
    ///
    /// Generally it is a bad idea to use negative target features because they often interact very
    /// poorly with how `-Ctarget-cpu` works. Instead, try to use a lower "base CPU" and enable the
    /// features you want to use.
    pub features: StaticCow<str>,
    /// Direct or use GOT indirect to reference external data symbols
    pub direct_access_external_data: Option<bool>,
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
    /// Also indicates whether to use Apple-specific ABI changes, such as extending function
    /// parameters to 32-bits.
    pub is_like_darwin: bool,
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
    /// Whether a target toolchain is like VEXos, the operating system used by the VEX Robotics V5 Brain.
    pub is_like_vexos: bool,
    /// Target's binary file format. Defaults to BinaryFormat::Elf
    pub binary_format: BinaryFormat,
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
    pub min_global_align: Option<Align>,

    /// Default number of codegen units to use in debug mode
    pub default_codegen_units: Option<u64>,

    /// Default codegen backend used for this target. Defaults to `None`.
    ///
    /// If `None`, then `CFG_DEFAULT_CODEGEN_BACKEND` environmental variable captured when
    /// compiling `rustc` will be used instead (or llvm if it is not set).
    ///
    /// N.B. when *using* the compiler, backend can always be overridden with `-Zcodegen-backend`.
    ///
    /// This was added by WaffleLapkin in #116793. The motivation is a rustc fork that requires a
    /// custom codegen backend for a particular target.
    pub default_codegen_backend: Option<StaticCow<str>>,

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

    /// The default visibility for symbols in this target.
    ///
    /// This value typically shouldn't be accessed directly, but through the
    /// `rustc_session::Session::default_visibility` method, which allows `rustc` users to override
    /// this setting using cmdline flags.
    pub default_visibility: Option<SymbolVisibility>,

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

    /// Use LLVM intrinsic for mcount function name
    pub llvm_mcount_intrinsic: Option<StaticCow<str>>,

    /// LLVM ABI name, corresponds to the '-mabi' parameter available in multilib C compilers
    /// and the `-target-abi` flag in llc. In the LLVM API this is `MCOptions.ABIName`.
    pub llvm_abiname: StaticCow<str>,

    /// Control the float ABI to use, for architectures that support it. The only architecture we
    /// currently use this for is ARM. Corresponds to the `-float-abi` flag in llc. In the LLVM API
    /// this is `FloatABIType`. (clang's `-mfloat-abi` is similar but more complicated since it
    /// can also affect the `soft-float` target feature.)
    ///
    /// If not provided, LLVM will infer the float ABI from the target triple (`llvm_target`).
    pub llvm_floatabi: Option<FloatAbi>,

    /// Picks a specific ABI for this target. This is *not* just for "Rust" ABI functions,
    /// it can also affect "C" ABI functions; the point is that this flag is interpreted by
    /// rustc and not forwarded to LLVM.
    /// So far, this is only used on x86.
    pub rustc_abi: Option<RustcAbi>,

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

    /// The ABI of the entry function.
    /// Default value is `CanonAbi::C`
    pub entry_abi: CanonAbi,

    /// Whether the target supports XRay instrumentation.
    pub supports_xray: bool,

    /// The default address space for this target. When using LLVM as a backend, most targets simply
    /// use LLVM's default address space (0). Some other targets, such as CHERI targets, use a
    /// custom default address space (in this specific case, `200`).
    pub default_address_space: rustc_abi::AddressSpace,

    /// Whether the targets supports -Z small-data-threshold
    small_data_threshold_support: SmallDataThresholdSupport,
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
        | LinkerFlavor::Llbc
        | LinkerFlavor::Ptx => {}
    }
}

fn add_link_args(link_args: &mut LinkArgs, flavor: LinkerFlavor, args: &[&'static str]) {
    add_link_args_iter(link_args, flavor, args.iter().copied().map(Cow::Borrowed))
}

impl TargetOptions {
    pub fn supports_comdat(&self) -> bool {
        // XCOFF and MachO don't support COMDAT.
        !self.is_like_aix && !self.is_like_darwin
    }
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
        self.linker_flavor_json = self.linker_flavor.to_cli_counterpart();
        self.lld_flavor_json = self.linker_flavor.lld_flavor();
        self.linker_is_gnu_json = self.linker_flavor.is_gnu();
        for (args, args_json) in [
            (&self.pre_link_args, &mut self.pre_link_args_json),
            (&self.late_link_args, &mut self.late_link_args_json),
            (&self.late_link_args_dynamic, &mut self.late_link_args_dynamic_json),
            (&self.late_link_args_static, &mut self.late_link_args_static_json),
            (&self.post_link_args, &mut self.post_link_args_json),
        ] {
            *args_json = args
                .iter()
                .map(|(flavor, args)| (flavor.to_cli_counterpart(), args.clone()))
                .collect();
        }
    }
}

impl Default for TargetOptions {
    /// Creates a set of "sane defaults" for any target. This is still
    /// incomplete, and if used for compilation, will certainly not work.
    fn default() -> TargetOptions {
        TargetOptions {
            endian: Endian::Little,
            c_int_width: 32,
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
            need_explicit_cpu: false,
            features: "".into(),
            direct_access_external_data: None,
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
            is_like_darwin: false,
            is_like_solaris: false,
            is_like_windows: false,
            is_like_msvc: false,
            is_like_wasm: false,
            is_like_android: false,
            is_like_vexos: false,
            binary_format: BinaryFormat::Elf,
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
            default_codegen_backend: None,
            trap_unreachable: true,
            requires_lto: false,
            singlethread: false,
            no_builtins: false,
            default_visibility: None,
            emit_debug_gdb_scripts: true,
            requires_uwtable: false,
            default_uwtable: false,
            simd_types_indirect: true,
            limit_rdylib_exports: true,
            override_export_symbols: None,
            merge_functions: MergeFunctions::Aliases,
            mcount: "mcount".into(),
            llvm_mcount_intrinsic: None,
            llvm_abiname: "".into(),
            llvm_floatabi: None,
            rustc_abi: None,
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
            c_enum_min_bits: None,
            generate_arange_section: true,
            supports_stack_protector: true,
            entry_name: "main".into(),
            entry_abi: CanonAbi::C,
            supports_xray: false,
            default_address_space: rustc_abi::AddressSpace::ZERO,
            small_data_threshold_support: SmallDataThresholdSupport::DefaultForArch,
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
    pub fn is_abi_supported(&self, abi: ExternAbi) -> bool {
        let abi_map = AbiMap::from_target(self);
        abi_map.canonize_abi(abi, false).is_mapped()
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

    /// Check some basic consistency of the current target. For JSON targets we are less strict;
    /// some of these checks are more guidelines than strict rules.
    fn check_consistency(&self, kind: TargetKind) -> Result<(), String> {
        macro_rules! check {
            ($b:expr, $($msg:tt)*) => {
                if !$b {
                    return Err(format!($($msg)*));
                }
            }
        }
        macro_rules! check_eq {
            ($left:expr, $right:expr, $($msg:tt)*) => {
                if ($left) != ($right) {
                    return Err(format!($($msg)*));
                }
            }
        }
        macro_rules! check_ne {
            ($left:expr, $right:expr, $($msg:tt)*) => {
                if ($left) == ($right) {
                    return Err(format!($($msg)*));
                }
            }
        }
        macro_rules! check_matches {
            ($left:expr, $right:pat, $($msg:tt)*) => {
                if !matches!($left, $right) {
                    return Err(format!($($msg)*));
                }
            }
        }

        check_eq!(
            self.is_like_darwin,
            self.vendor == "apple",
            "`is_like_darwin` must be set if and only if `vendor` is `apple`"
        );
        check_eq!(
            self.is_like_solaris,
            self.os == "solaris" || self.os == "illumos",
            "`is_like_solaris` must be set if and only if `os` is `solaris` or `illumos`"
        );
        check_eq!(
            self.is_like_windows,
            self.os == "windows" || self.os == "uefi" || self.os == "cygwin",
            "`is_like_windows` must be set if and only if `os` is `windows`, `uefi` or `cygwin`"
        );
        check_eq!(
            self.is_like_wasm,
            self.arch == "wasm32" || self.arch == "wasm64",
            "`is_like_wasm` must be set if and only if `arch` is `wasm32` or `wasm64`"
        );
        if self.is_like_msvc {
            check!(self.is_like_windows, "if `is_like_msvc` is set, `is_like_windows` must be set");
        }
        if self.os == "emscripten" {
            check!(self.is_like_wasm, "the `emcscripten` os only makes sense on wasm-like targets");
        }

        // Check that default linker flavor is compatible with some other key properties.
        check_eq!(
            self.is_like_darwin,
            matches!(self.linker_flavor, LinkerFlavor::Darwin(..)),
            "`linker_flavor` must be `darwin` if and only if `is_like_darwin` is set"
        );
        check_eq!(
            self.is_like_msvc,
            matches!(self.linker_flavor, LinkerFlavor::Msvc(..)),
            "`linker_flavor` must be `msvc` if and only if `is_like_msvc` is set"
        );
        check_eq!(
            self.is_like_wasm && self.os != "emscripten",
            matches!(self.linker_flavor, LinkerFlavor::WasmLld(..)),
            "`linker_flavor` must be `wasm-lld` if and only if `is_like_wasm` is set and the `os` is not `emscripten`",
        );
        check_eq!(
            self.os == "emscripten",
            matches!(self.linker_flavor, LinkerFlavor::EmCc),
            "`linker_flavor` must be `em-cc` if and only if `os` is `emscripten`"
        );
        check_eq!(
            self.arch == "bpf",
            matches!(self.linker_flavor, LinkerFlavor::Bpf),
            "`linker_flavor` must be `bpf` if and only if `arch` is `bpf`"
        );
        check_eq!(
            self.arch == "nvptx64",
            matches!(self.linker_flavor, LinkerFlavor::Ptx),
            "`linker_flavor` must be `ptc` if and only if `arch` is `nvptx64`"
        );

        for args in [
            &self.pre_link_args,
            &self.late_link_args,
            &self.late_link_args_dynamic,
            &self.late_link_args_static,
            &self.post_link_args,
        ] {
            for (&flavor, flavor_args) in args {
                check!(
                    !flavor_args.is_empty() || self.arch == "avr",
                    "linker flavor args must not be empty"
                );
                // Check that flavors mentioned in link args are compatible with the default flavor.
                match self.linker_flavor {
                    LinkerFlavor::Gnu(..) => {
                        check_matches!(
                            flavor,
                            LinkerFlavor::Gnu(..),
                            "mixing GNU and non-GNU linker flavors"
                        );
                    }
                    LinkerFlavor::Darwin(..) => {
                        check_matches!(
                            flavor,
                            LinkerFlavor::Darwin(..),
                            "mixing Darwin and non-Darwin linker flavors"
                        )
                    }
                    LinkerFlavor::WasmLld(..) => {
                        check_matches!(
                            flavor,
                            LinkerFlavor::WasmLld(..),
                            "mixing wasm and non-wasm linker flavors"
                        )
                    }
                    LinkerFlavor::Unix(..) => {
                        check_matches!(
                            flavor,
                            LinkerFlavor::Unix(..),
                            "mixing unix and non-unix linker flavors"
                        );
                    }
                    LinkerFlavor::Msvc(..) => {
                        check_matches!(
                            flavor,
                            LinkerFlavor::Msvc(..),
                            "mixing MSVC and non-MSVC linker flavors"
                        );
                    }
                    LinkerFlavor::EmCc
                    | LinkerFlavor::Bpf
                    | LinkerFlavor::Ptx
                    | LinkerFlavor::Llbc => {
                        check_eq!(flavor, self.linker_flavor, "mixing different linker flavors")
                    }
                }

                // Check that link args for cc and non-cc versions of flavors are consistent.
                let check_noncc = |noncc_flavor| -> Result<(), String> {
                    if let Some(noncc_args) = args.get(&noncc_flavor) {
                        for arg in flavor_args {
                            if let Some(suffix) = arg.strip_prefix("-Wl,") {
                                check!(
                                    noncc_args.iter().any(|a| a == suffix),
                                    " link args for cc and non-cc versions of flavors are not consistent"
                                );
                            }
                        }
                    }
                    Ok(())
                };

                match self.linker_flavor {
                    LinkerFlavor::Gnu(Cc::Yes, lld) => check_noncc(LinkerFlavor::Gnu(Cc::No, lld))?,
                    LinkerFlavor::WasmLld(Cc::Yes) => check_noncc(LinkerFlavor::WasmLld(Cc::No))?,
                    LinkerFlavor::Unix(Cc::Yes) => check_noncc(LinkerFlavor::Unix(Cc::No))?,
                    _ => {}
                }
            }

            // Check that link args for lld and non-lld versions of flavors are consistent.
            for cc in [Cc::No, Cc::Yes] {
                check_eq!(
                    args.get(&LinkerFlavor::Gnu(cc, Lld::No)),
                    args.get(&LinkerFlavor::Gnu(cc, Lld::Yes)),
                    "link args for lld and non-lld versions of flavors are not consistent",
                );
                check_eq!(
                    args.get(&LinkerFlavor::Darwin(cc, Lld::No)),
                    args.get(&LinkerFlavor::Darwin(cc, Lld::Yes)),
                    "link args for lld and non-lld versions of flavors are not consistent",
                );
            }
            check_eq!(
                args.get(&LinkerFlavor::Msvc(Lld::No)),
                args.get(&LinkerFlavor::Msvc(Lld::Yes)),
                "link args for lld and non-lld versions of flavors are not consistent",
            );
        }

        if self.link_self_contained.is_disabled() {
            check!(
                self.pre_link_objects_self_contained.is_empty()
                    && self.post_link_objects_self_contained.is_empty(),
                "if `link_self_contained` is disabled, then `pre_link_objects_self_contained` and `post_link_objects_self_contained` must be empty",
            );
        }

        // If your target really needs to deviate from the rules below,
        // except it and document the reasons.
        // Keep the default "unknown" vendor instead.
        check_ne!(self.vendor, "", "`vendor` cannot be empty");
        check_ne!(self.os, "", "`os` cannot be empty");
        if !self.can_use_os_unknown() {
            // Keep the default "none" for bare metal targets instead.
            check_ne!(
                self.os,
                "unknown",
                "`unknown` os can only be used on particular targets; use `none` for bare-metal targets"
            );
        }

        // Check dynamic linking stuff.
        // We skip this for JSON targets since otherwise, our default values would fail this test.
        // These checks are not critical for correctness, but more like default guidelines.
        // FIXME (https://github.com/rust-lang/rust/issues/133459): do we want to change the JSON
        // target defaults so that they pass these checks?
        if kind == TargetKind::Builtin {
            // BPF: when targeting user space vms (like rbpf), those can load dynamic libraries.
            // hexagon: when targeting QuRT, that OS can load dynamic libraries.
            // wasm{32,64}: dynamic linking is inherent in the definition of the VM.
            if self.os == "none"
                && (self.arch != "bpf"
                    && self.arch != "hexagon"
                    && self.arch != "wasm32"
                    && self.arch != "wasm64")
            {
                check!(
                    !self.dynamic_linking,
                    "dynamic linking is not supported on this OS/architecture"
                );
            }
            if self.only_cdylib
                || self.crt_static_allows_dylibs
                || !self.late_link_args_dynamic.is_empty()
            {
                check!(
                    self.dynamic_linking,
                    "dynamic linking must be allowed when `only_cdylib` or `crt_static_allows_dylibs` or `late_link_args_dynamic` are set"
                );
            }
            // Apparently PIC was slow on wasm at some point, see comments in wasm_base.rs
            if self.dynamic_linking && !self.is_like_wasm {
                check_eq!(
                    self.relocation_model,
                    RelocModel::Pic,
                    "targets that support dynamic linking must use the `pic` relocation model"
                );
            }
            if self.position_independent_executables {
                check_eq!(
                    self.relocation_model,
                    RelocModel::Pic,
                    "targets that support position-independent executables must use the `pic` relocation model"
                );
            }
            // The UEFI targets do not support dynamic linking but still require PIC (#101377).
            if self.relocation_model == RelocModel::Pic && (self.os != "uefi") {
                check!(
                    self.dynamic_linking || self.position_independent_executables,
                    "when the relocation model is `pic`, the target must support dynamic linking or use position-independent executables. \
                Set the relocation model to `static` to avoid this requirement"
                );
            }
            if self.static_position_independent_executables {
                check!(
                    self.position_independent_executables,
                    "if `static_position_independent_executables` is set, then `position_independent_executables` must be set"
                );
            }
            if self.position_independent_executables {
                check!(
                    self.executables,
                    "if `position_independent_executables` is set then `executables` must be set"
                );
            }
        }

        // Check crt static stuff
        if self.crt_static_default || self.crt_static_allows_dylibs {
            check!(
                self.crt_static_respected,
                "static CRT can be enabled but `crt_static_respected` is not set"
            );
        }

        // Check that RISC-V targets always specify which ABI they use,
        // and that ARM targets specify their float ABI.
        match &*self.arch {
            "riscv32" => {
                check_matches!(
                    &*self.llvm_abiname,
                    "ilp32" | "ilp32f" | "ilp32d" | "ilp32e",
                    "invalid RISC-V ABI name: {}",
                    self.llvm_abiname,
                );
            }
            "riscv64" => {
                // Note that the `lp64e` is still unstable as it's not (yet) part of the ELF psABI.
                check_matches!(
                    &*self.llvm_abiname,
                    "lp64" | "lp64f" | "lp64d" | "lp64e",
                    "invalid RISC-V ABI name: {}",
                    self.llvm_abiname,
                );
            }
            "arm" => {
                check!(
                    self.llvm_floatabi.is_some(),
                    "ARM targets must set `llvm-floatabi` to `hard` or `soft`",
                )
            }
            _ => {}
        }

        // Check consistency of Rust ABI declaration.
        if let Some(rust_abi) = self.rustc_abi {
            match rust_abi {
                RustcAbi::X86Sse2 => check_matches!(
                    &*self.arch,
                    "x86",
                    "`x86-sse2` ABI is only valid for x86-32 targets"
                ),
                RustcAbi::X86Softfloat => check_matches!(
                    &*self.arch,
                    "x86" | "x86_64",
                    "`x86-softfloat` ABI is only valid for x86 targets"
                ),
            }
        }

        // Check that the given target-features string makes some basic sense.
        if !self.features.is_empty() {
            let mut features_enabled = FxHashSet::default();
            let mut features_disabled = FxHashSet::default();
            for feat in self.features.split(',') {
                if let Some(feat) = feat.strip_prefix("+") {
                    features_enabled.insert(feat);
                    if features_disabled.contains(feat) {
                        return Err(format!(
                            "target feature `{feat}` is both enabled and disabled"
                        ));
                    }
                } else if let Some(feat) = feat.strip_prefix("-") {
                    features_disabled.insert(feat);
                    if features_enabled.contains(feat) {
                        return Err(format!(
                            "target feature `{feat}` is both enabled and disabled"
                        ));
                    }
                } else {
                    return Err(format!(
                        "target feature `{feat}` is invalid, must start with `+` or `-`"
                    ));
                }
            }
            // Check that we don't mis-set any of the ABI-relevant features.
            let abi_feature_constraints = self.abi_required_features();
            for feat in abi_feature_constraints.required {
                // The feature might be enabled by default so we can't *require* it to show up.
                // But it must not be *disabled*.
                if features_disabled.contains(feat) {
                    return Err(format!(
                        "target feature `{feat}` is required by the ABI but gets disabled in target spec"
                    ));
                }
            }
            for feat in abi_feature_constraints.incompatible {
                // The feature might be disabled by default so we can't *require* it to show up.
                // But it must not be *enabled*.
                if features_enabled.contains(feat) {
                    return Err(format!(
                        "target feature `{feat}` is incompatible with the ABI but gets enabled in target spec"
                    ));
                }
            }
        }

        Ok(())
    }

    /// Test target self-consistency and JSON encoding/decoding roundtrip.
    #[cfg(test)]
    fn test_target(mut self) {
        let recycled_target =
            Target::from_json(&serde_json::to_string(&self.to_json()).unwrap()).map(|(j, _)| j);
        self.update_to_cli();
        self.check_consistency(TargetKind::Builtin).unwrap();
        assert_eq!(recycled_target, Ok(self));
    }

    // Add your target to the whitelist if it has `std` library
    // and you certainly want "unknown" for the OS name.
    fn can_use_os_unknown(&self) -> bool {
        self.llvm_target == "wasm32-unknown-unknown"
            || self.llvm_target == "wasm64-unknown-unknown"
            || (self.env == "sgx" && self.vendor == "fortanix")
    }

    /// Load a built-in target
    pub fn expect_builtin(target_tuple: &TargetTuple) -> Target {
        match *target_tuple {
            TargetTuple::TargetTuple(ref target_tuple) => {
                load_builtin(target_tuple).expect("built-in target")
            }
            TargetTuple::TargetJson { .. } => {
                panic!("built-in targets doesn't support target-paths")
            }
        }
    }

    /// Load all built-in targets
    pub fn builtins() -> impl Iterator<Item = Target> {
        load_all_builtins()
    }

    /// Search for a JSON file specifying the given target tuple.
    ///
    /// If none is found in `$RUST_TARGET_PATH`, look for a file called `target.json` inside the
    /// sysroot under the target-tuple's `rustlib` directory. Note that it could also just be a
    /// bare filename already, so also check for that. If one of the hardcoded targets we know
    /// about, just return it directly.
    ///
    /// The error string could come from any of the APIs called, including filesystem access and
    /// JSON decoding.
    pub fn search(
        target_tuple: &TargetTuple,
        sysroot: &Path,
    ) -> Result<(Target, TargetWarnings), String> {
        use std::{env, fs};

        fn load_file(path: &Path) -> Result<(Target, TargetWarnings), String> {
            let contents = fs::read_to_string(path).map_err(|e| e.to_string())?;
            Target::from_json(&contents)
        }

        match *target_tuple {
            TargetTuple::TargetTuple(ref target_tuple) => {
                // check if tuple is in list of built-in targets
                if let Some(t) = load_builtin(target_tuple) {
                    return Ok((t, TargetWarnings::empty()));
                }

                // search for a file named `target_tuple`.json in RUST_TARGET_PATH
                let path = {
                    let mut target = target_tuple.to_string();
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

                // Additionally look in the sysroot under `lib/rustlib/<tuple>/target.json`
                // as a fallback.
                let rustlib_path = crate::relative_target_rustlib_path(sysroot, target_tuple);
                let p = PathBuf::from_iter([
                    Path::new(sysroot),
                    Path::new(&rustlib_path),
                    Path::new("target.json"),
                ]);
                if p.is_file() {
                    return load_file(&p);
                }

                // Leave in a specialized error message for the removed target.
                // FIXME: If you see this and it's been a few months after this has been released,
                // you can probably remove it.
                if target_tuple == "i586-pc-windows-msvc" {
                    Err("the `i586-pc-windows-msvc` target has been removed. Use the `i686-pc-windows-msvc` target instead.\n\
                        Windows 10 (the minimum required OS version) requires a CPU baseline of at least i686 so you can safely switch".into())
                } else {
                    Err(format!("could not find specification for target {target_tuple:?}"))
                }
            }
            TargetTuple::TargetJson { ref contents, .. } => Target::from_json(contents),
        }
    }

    /// Return the target's small data threshold support, converting
    /// `DefaultForArch` into a concrete value.
    pub fn small_data_threshold_support(&self) -> SmallDataThresholdSupport {
        match &self.options.small_data_threshold_support {
            // Avoid having to duplicate the small data support in every
            // target file by supporting a default value for each
            // architecture.
            SmallDataThresholdSupport::DefaultForArch => match self.arch.as_ref() {
                "mips" | "mips64" | "mips32r6" => {
                    SmallDataThresholdSupport::LlvmArg("mips-ssection-threshold".into())
                }
                "hexagon" => {
                    SmallDataThresholdSupport::LlvmArg("hexagon-small-data-threshold".into())
                }
                "m68k" => SmallDataThresholdSupport::LlvmArg("m68k-ssection-threshold".into()),
                "riscv32" | "riscv64" => {
                    SmallDataThresholdSupport::LlvmModuleFlag("SmallDataLimit".into())
                }
                _ => SmallDataThresholdSupport::None,
            },
            s => s.clone(),
        }
    }

    pub fn object_architecture(
        &self,
        unstable_target_features: &FxIndexSet<Symbol>,
    ) -> Option<(object::Architecture, Option<object::SubArchitecture>)> {
        use object::Architecture;
        Some(match self.arch.as_ref() {
            "arm" => (Architecture::Arm, None),
            "aarch64" => (
                if self.pointer_width == 32 {
                    Architecture::Aarch64_Ilp32
                } else {
                    Architecture::Aarch64
                },
                None,
            ),
            "x86" => (Architecture::I386, None),
            "s390x" => (Architecture::S390x, None),
            "m68k" => (Architecture::M68k, None),
            "mips" | "mips32r6" => (Architecture::Mips, None),
            "mips64" | "mips64r6" => (
                // While there are currently no builtin targets
                // using the N32 ABI, it is possible to specify
                // it using a custom target specification. N32
                // is an ILP32 ABI like the Aarch64_Ilp32
                // and X86_64_X32 cases above and below this one.
                if self.options.llvm_abiname.as_ref() == "n32" {
                    Architecture::Mips64_N32
                } else {
                    Architecture::Mips64
                },
                None,
            ),
            "x86_64" => (
                if self.pointer_width == 32 {
                    Architecture::X86_64_X32
                } else {
                    Architecture::X86_64
                },
                None,
            ),
            "powerpc" => (Architecture::PowerPc, None),
            "powerpc64" => (Architecture::PowerPc64, None),
            "riscv32" => (Architecture::Riscv32, None),
            "riscv64" => (Architecture::Riscv64, None),
            "sparc" => {
                if unstable_target_features.contains(&sym::v8plus) {
                    // Target uses V8+, aka EM_SPARC32PLUS, aka 64-bit V9 but in 32-bit mode
                    (Architecture::Sparc32Plus, None)
                } else {
                    // Target uses V7 or V8, aka EM_SPARC
                    (Architecture::Sparc, None)
                }
            }
            "sparc64" => (Architecture::Sparc64, None),
            "avr" => (Architecture::Avr, None),
            "msp430" => (Architecture::Msp430, None),
            "hexagon" => (Architecture::Hexagon, None),
            "xtensa" => (Architecture::Xtensa, None),
            "bpf" => (Architecture::Bpf, None),
            "loongarch32" => (Architecture::LoongArch32, None),
            "loongarch64" => (Architecture::LoongArch64, None),
            "csky" => (Architecture::Csky, None),
            "arm64ec" => (Architecture::Aarch64, Some(object::SubArchitecture::Arm64EC)),
            // Unsupported architecture.
            _ => return None,
        })
    }

    /// Returns whether this target is known to have unreliable alignment:
    /// native C code for the target fails to align some data to the degree
    /// required by the C standard. We can't *really* do anything about that
    /// since unsafe Rust code may assume alignment any time, but we can at least
    /// inhibit some optimizations, and we suppress the alignment checks that
    /// would detect this unsoundness.
    ///
    /// Every target that returns less than `Align::MAX` here is still has a soundness bug.
    pub fn max_reliable_alignment(&self) -> Align {
        // FIXME(#112480) MSVC on x86-32 is unsound and fails to properly align many types with
        // more-than-4-byte-alignment on the stack. This makes alignments larger than 4 generally
        // unreliable on 32bit Windows.
        if self.is_like_windows && self.arch == "x86" {
            Align::from_bytes(4).unwrap()
        } else {
            Align::MAX
        }
    }
}

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
