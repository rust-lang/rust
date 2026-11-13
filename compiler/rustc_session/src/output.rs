//! Related to out filenames of compilation (e.g. binaries).
use std::collections::BTreeMap;
use std::collections::btree_map::{
    Iter as BTreeMapIter, Keys as BTreeMapKeysIter, Values as BTreeMapValuesIter,
};
use std::path::Path;

use rustc_data_structures::stable_hash::StableOrd;
use rustc_errors::ColorConfig;
use rustc_errors::emitter::HumanReadableErrorType;
use rustc_macros::{Decodable, Encodable, StableHash};
use rustc_span::{Span, Symbol};
use rustc_structures::CrateType;

use crate::Session;
use crate::config::{OutFileName, OutputFilenames};
use crate::diagnostics::{
    CrateNameEmpty, FileIsNotWriteable, InvalidCharacterInCrateName,
    InvalidCharacterInCrateNameSuggestion,
};

pub fn out_filename(
    sess: &Session,
    crate_type: CrateType,
    outputs: &OutputFilenames,
    crate_name: Symbol,
) -> OutFileName {
    let default_filename = filename_for_input(sess, crate_type, crate_name, outputs);
    let out_filename = outputs
        .outputs
        .get(&OutputType::Exe)
        .and_then(|s| s.to_owned())
        .or_else(|| outputs.single_output_file.clone())
        .unwrap_or(default_filename);

    if let OutFileName::Real(ref path) = out_filename {
        check_file_is_writeable(path, sess);
    }

    out_filename
}

/// Make sure files are writeable. Mac, FreeBSD, and Windows system linkers
/// check this already -- however, the Linux linker will happily overwrite a
/// read-only file. We should be consistent.
pub fn check_file_is_writeable(file: &Path, sess: &Session) {
    if !is_writeable(file) {
        sess.dcx().emit_fatal(FileIsNotWriteable { file });
    }
}

fn is_writeable(p: &Path) -> bool {
    match p.metadata() {
        Err(..) => true,
        Ok(m) => !m.permissions().readonly(),
    }
}

/// Validate the given crate name.
///
/// Note that this validation is more permissive than identifier parsing. It considers
/// non-empty sequences of alphanumeric and underscore characters to be valid crate names.
/// Most notably, it accepts names starting with a numeric character like `0`!
///
/// Furthermore, this shouldn't be taken as the canonical crate name validator.
/// Other places may use a more restrictive grammar (e.g., identifier or ASCII identifier).
pub fn validate_crate_name(sess: &Session, crate_name: Symbol, span: Option<Span>) {
    let mut guar = None;

    if crate_name.is_empty() {
        guar = Some(sess.dcx().emit_err(CrateNameEmpty { span }));
    }

    for c in crate_name.as_str().chars() {
        if c.is_alphanumeric() || c == '_' {
            continue;
        }
        guar = Some(sess.dcx().emit_err(InvalidCharacterInCrateName {
            span,
            character: c,
            crate_name,
            suggestion: span.is_none().then(|| InvalidCharacterInCrateNameSuggestion {
                suggested_name:
                    crate_name.as_str().replace(|c: char| c != '_' && !c.is_alphanumeric(), "_"),
            }),
        }));
    }

    if let Some(guar) = guar {
        guar.raise_fatal();
    }
}

pub fn filename_for_metadata(sess: &Session, outputs: &OutputFilenames) -> OutFileName {
    let output_type = sess.opts.output_types.metadata_output_type();
    let out_filename = outputs.path(output_type);
    if let OutFileName::Real(ref path) = out_filename {
        check_file_is_writeable(path, sess);
    }
    out_filename
}

pub fn filename_for_input(
    sess: &Session,
    crate_type: CrateType,
    crate_name: Symbol,
    outputs: &OutputFilenames,
) -> OutFileName {
    let libname = format!("{}{}", crate_name, sess.opts.cg.extra_filename);

    match crate_type {
        CrateType::Rlib => {
            OutFileName::Real(outputs.out_directory.join(&format!("lib{libname}.rlib")))
        }
        CrateType::Cdylib | CrateType::ProcMacro | CrateType::Dylib | CrateType::Sdylib => {
            let (prefix, suffix) = (&sess.target.dll_prefix, &sess.target.dll_suffix);
            OutFileName::Real(outputs.out_directory.join(&format!("{prefix}{libname}{suffix}")))
        }
        CrateType::StaticLib => {
            let (prefix, suffix) = sess.staticlib_components(false);
            OutFileName::Real(outputs.out_directory.join(&format!("{prefix}{libname}{suffix}")))
        }
        CrateType::Executable => {
            let suffix = &sess.target.exe_suffix;
            let out_filename = outputs.path(OutputType::Exe);
            if let OutFileName::Real(ref path) = out_filename {
                if suffix.is_empty() {
                    out_filename
                } else {
                    OutFileName::Real(path.with_extension(&suffix[1..]))
                }
            } else {
                out_filename
            }
        }
    }
}

/// Checks if target supports crate_type as output
pub fn invalid_output_for_target(sess: &Session, crate_type: CrateType) -> bool {
    if let CrateType::Cdylib | CrateType::Dylib | CrateType::ProcMacro = crate_type {
        if !sess.target.dynamic_linking {
            return true;
        }
        if sess.crt_static(Some(crate_type)) && !sess.target.crt_static_allows_dylibs {
            return true;
        }
    }
    if crate_type == CrateType::ProcMacro {
        if sess.opts.target_triple == sess.wasm_proc_macro_tuple
            && sess.opts.unstable_opts.wasm_proc_macros
        {
            return false;
        }
    }
    if let CrateType::ProcMacro | CrateType::Dylib = crate_type
        && sess.target.only_cdylib
    {
        return true;
    }
    if let CrateType::Executable = crate_type
        && !sess.target.executables
    {
        return true;
    }

    false
}

macro_rules! define_output_types {
    (
        $(
            $(#[doc = $doc:expr])*
            $Variant:ident => {
                shorthand: $shorthand:expr,
                extension: $extension:expr,
                description: $description:expr,
                default_filename: $default_filename:expr,
                is_text: $is_text:expr,
                compatible_with_cgus_and_single_output: $compatible:expr
            }
        ),* $(,)?
    ) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord, StableHash)]
        #[derive(Encodable, Decodable)]
        pub enum OutputType {
            $(
                $(#[doc = $doc])*
                $Variant,
            )*
        }

        impl StableOrd for OutputType {
            const CAN_USE_UNSTABLE_SORT: bool = true;

            // Trivial C-Style enums have a stable sort order across compilation sessions.
            const THIS_IMPLEMENTATION_HAS_BEEN_TRIPLE_CHECKED: () = ();
        }

        impl OutputType {
            pub fn iter_all() -> impl Iterator<Item = OutputType> {
                static ALL_VARIANTS: &[OutputType] = &[
                    $(
                        OutputType::$Variant,
                    )*
                ];
                ALL_VARIANTS.iter().copied()
            }

            pub(crate) fn is_compatible_with_codegen_units_and_single_output_file(&self) -> bool {
                match *self {
                    $(
                        OutputType::$Variant => $compatible,
                    )*
                }
            }

            pub fn shorthand(&self) -> &'static str {
                match *self {
                    $(
                        OutputType::$Variant => $shorthand,
                    )*
                }
            }

            pub(crate) fn from_shorthand(shorthand: &str) -> Option<Self> {
                match shorthand {
                    $(
                        s if s == $shorthand => Some(OutputType::$Variant),
                    )*
                    _ => None,
                }
            }

            pub(crate) fn shorthands_display() -> String {
                let shorthands = vec![
                    $(
                        format!("`{}`", $shorthand),
                    )*
                ];
                shorthands.join(", ")
            }

            pub fn extension(&self) -> &'static str {
                match *self {
                    $(
                        OutputType::$Variant => $extension,
                    )*
                }
            }

            pub fn is_text_output(&self) -> bool {
                match *self {
                    $(
                        OutputType::$Variant => $is_text,
                    )*
                }
            }

            pub fn description(&self) -> &'static str {
                match *self {
                    $(
                        OutputType::$Variant => $description,
                    )*
                }
            }

            pub fn default_filename(&self) -> &'static str {
                match *self {
                    $(
                        OutputType::$Variant => $default_filename,
                    )*
                }
            }


        }
    }
}

define_output_types! {
    Validation => {
        shorthand: "validation",
        extension: "validation.rmeta",
        description: "Generates metadata and exits after doing only validation and required analysis",
        default_filename: "CRATE_NAME.validation.rmeta",
        is_text: false,
        compatible_with_cgus_and_single_output: true
    },
    DepInfo => {
        shorthand: "dep-info",
        extension: "d",
        description: "Generates a file with Makefile syntax that indicates all the source files that were loaded to generate the crate",
        default_filename: "CRATE_NAME.d",
        is_text: true,
        compatible_with_cgus_and_single_output: true
    },
    FullAnalysis => {
        shorthand: "full-analysis",
        extension: "full-analysis.rmeta",
        description: "Generates metadata and exits after doing all analysis",
        default_filename: "CRATE_NAME.full-analysis.rmeta",
        is_text: false,
        compatible_with_cgus_and_single_output: true
    },
    Mir => {
        shorthand: "mir",
        extension: "mir",
        description: "Generates a file containing rustc's mid-level intermediate representation",
        default_filename: "CRATE_NAME.mir",
        is_text: true,
        compatible_with_cgus_and_single_output: false
    },
    Metadata => {
        shorthand: "metadata",
        extension: "rmeta",
        description: "Generates a file containing metadata about the crate",
        default_filename: "libCRATE_NAME.rmeta",
        is_text: false,
        compatible_with_cgus_and_single_output: true
    },
    #[doc = "This is the summary or index data part of the ThinLTO bitcode."]
    ThinLinkBitcode => {
        shorthand: "thin-link-bitcode",
        extension: "indexing.o",
        description: "Generates the ThinLTO summary as bitcode",
        default_filename: "CRATE_NAME.indexing.o",
        is_text: false,
        compatible_with_cgus_and_single_output: false
    },
    LlvmAssembly => {
        shorthand: "llvm-ir",
        extension: "ll",
        description: "Generates a file containing LLVM IR",
        default_filename: "CRATE_NAME.ll",
        is_text: true,
        compatible_with_cgus_and_single_output: false
    },
    #[doc = "This is the optimized bitcode, which could be either pre-LTO or non-LTO bitcode,"]
    #[doc = "depending on the specific request type."]
    Bitcode => {
        shorthand: "llvm-bc",
        extension: "bc",
        description: "Generates a binary file containing the LLVM bitcode",
        default_filename: "CRATE_NAME.bc",
        is_text: false,
        compatible_with_cgus_and_single_output: false
    },
    Assembly => {
        shorthand: "asm",
        extension: "s",
        description: "Generates a file with the crate's assembly code",
        default_filename: "CRATE_NAME.s",
        is_text: true,
        compatible_with_cgus_and_single_output: false
    },
    Object => {
        shorthand: "obj",
        extension: "o",
        description: "Generates a native object file",
        default_filename: "CRATE_NAME.o",
        is_text: false,
        compatible_with_cgus_and_single_output: false
    },
    Exe => {
        shorthand: "link",
        extension: "",
        description: "Generates the crates specified by --crate-type. This is the default if --emit is not specified",
        default_filename: "(platform and crate-type dependent)",
        is_text: false,
        compatible_with_cgus_and_single_output: true
    },
}

/// The type of diagnostics output to generate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ErrorOutputType {
    /// Output meant for the consumption of humans.
    #[default]
    HumanReadable {
        kind: HumanReadableErrorType = HumanReadableErrorType { no_graphics: false, short: false, unicode: false },
        color_config: ColorConfig = ColorConfig::Auto,
    },
    /// Output that's consumed by other tools such as `rustfix` or the `RLS`.
    Json {
        /// Render the JSON in a human readable way (with indents and newlines).
        pretty: bool,
        /// The JSON output includes a `rendered` field that includes the rendered
        /// human output.
        json_rendered: HumanReadableErrorType,
        color_config: ColorConfig,
    },
}

#[derive(Clone, Hash, Debug)]
pub enum ResolveDocLinks {
    /// Do not resolve doc links.
    None,
    /// Resolve doc links on exported items only for crate types that have metadata.
    ExportedMetadata,
    /// Resolve doc links on exported items.
    Exported,
    /// Resolve doc links on all items.
    All,
}

/// Use tree-based collections to cheaply get a deterministic `Hash` implementation.
/// *Do not* switch `BTreeMap` out for an unsorted container type! That would break
/// dependency tracking for command-line arguments. Also only hash keys, since tracking
/// should only depend on the output types, not the paths they're written to.
#[derive(Clone, Debug, Hash, StableHash, Encodable, Decodable)]
pub struct OutputTypes(pub(crate) BTreeMap<OutputType, Option<OutFileName>>);

impl OutputTypes {
    pub fn new(entries: &[(OutputType, Option<OutFileName>)]) -> OutputTypes {
        OutputTypes(BTreeMap::from_iter(entries.iter().map(|&(k, ref v)| (k, v.clone()))))
    }

    pub(crate) fn get(&self, key: &OutputType) -> Option<&Option<OutFileName>> {
        self.0.get(key)
    }

    pub fn contains_key(&self, key: &OutputType) -> bool {
        self.0.contains_key(key)
    }

    /// Returns `true` if user specified a name and not just produced type
    pub fn contains_explicit_name(&self, key: &OutputType) -> bool {
        matches!(self.0.get(key), Some(Some(..)))
    }

    pub fn iter(&self) -> BTreeMapIter<'_, OutputType, Option<OutFileName>> {
        self.0.iter()
    }

    pub fn keys(&self) -> BTreeMapKeysIter<'_, OutputType, Option<OutFileName>> {
        self.0.keys()
    }

    pub fn values(&self) -> BTreeMapValuesIter<'_, OutputType, Option<OutFileName>> {
        self.0.values()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if any of the output types require codegen or linking.
    pub fn should_codegen(&self) -> bool {
        self.0.keys().any(|k| match *k {
            OutputType::Bitcode
            | OutputType::ThinLinkBitcode
            | OutputType::Assembly
            | OutputType::LlvmAssembly
            | OutputType::Mir
            | OutputType::Object
            | OutputType::Exe => true,
            OutputType::Metadata
            | OutputType::DepInfo
            | OutputType::Validation
            | OutputType::FullAnalysis => false,
        })
    }

    /// Returns `true` if any of the output types require linking.
    pub fn should_link(&self) -> bool {
        self.0.keys().any(|k| match *k {
            OutputType::Bitcode
            | OutputType::ThinLinkBitcode
            | OutputType::Assembly
            | OutputType::LlvmAssembly
            | OutputType::Mir
            | OutputType::Metadata
            | OutputType::Object
            | OutputType::Validation
            | OutputType::FullAnalysis
            | OutputType::DepInfo => false,
            OutputType::Exe => true,
        })
    }

    pub fn needs_full_analysis(&self) -> bool {
        self.0.keys().any(|k| match *k {
            OutputType::Validation | OutputType::DepInfo => false,
            OutputType::Exe
            | OutputType::Bitcode
            | OutputType::Object
            | OutputType::Mir
            | OutputType::Assembly
            | OutputType::ThinLinkBitcode
            | OutputType::LlvmAssembly
            // FIXME: Metadata does not technically need it,
            // this is just to keep behaviour
            | OutputType::Metadata
            | OutputType::FullAnalysis => true,
        })
    }

    pub fn needs_codegen(&self) -> bool {
        self.0.keys().any(|k| match *k {
            OutputType::Validation
            | OutputType::DepInfo
            | OutputType::FullAnalysis => false,
            OutputType::Exe
            | OutputType::Bitcode
            | OutputType::Object
            | OutputType::Assembly
            | OutputType::ThinLinkBitcode
            | OutputType::LlvmAssembly
            | OutputType::Mir
            // FIXME: Metadata does not technically need it,
            // this is just to keep behaviour
            | OutputType::Metadata => true,
        })
    }

    pub fn needs_required_analysis(&self) -> bool {
        self.0.keys().any(|k| match *k {
            OutputType::DepInfo => false,
            OutputType::Exe
            | OutputType::Bitcode
            | OutputType::Object
            | OutputType::Assembly
            | OutputType::ThinLinkBitcode
            | OutputType::LlvmAssembly
            | OutputType::Mir
            | OutputType::FullAnalysis
            | OutputType::Validation
            // FIXME: Metadata does not technically need it,
            // this is just to keep behaviour
            | OutputType::Metadata => true,
        })
    }

    pub fn metadata_output_type(&self) -> OutputType {
        self.iter()
            .fold(None, |acc, (&out, _)| match (acc, out) {
                (Some(OutputType::Validation), OutputType::Metadata | OutputType::FullAnalysis) => {
                    Some(out)
                }
                (Some(OutputType::Validation), _) => acc,
                (Some(OutputType::FullAnalysis), OutputType::Metadata) => Some(out),
                (Some(OutputType::FullAnalysis), _) => acc,
                (Some(OutputType::Metadata), _) => Some(OutputType::Metadata),
                _ => Some(out),
            })
            .unwrap_or(OutputType::Metadata)
    }

    pub fn needs_metadata(&self) -> bool {
        self.iter().any(|(k, _)| *k == OutputType::Metadata)
    }

    pub fn needs_mir(&self) -> bool {
        self.0.keys().any(|k| k == &OutputType::Mir)
    }

    pub fn all<F: Fn(&OutputType) -> bool>(&self, f: F) -> bool {
        self.iter().map(|e| e.0).all(f)
    }
}
