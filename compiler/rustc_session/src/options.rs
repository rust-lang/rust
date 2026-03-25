use std::collections::BTreeMap;
use std::num::{IntErrorKind, NonZero};
use std::path::PathBuf;
use std::str;

use rustc_abi::Align;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::profiling::TimePassesFormat;
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_errors::{ColorConfig, TerminalUrl};
use rustc_feature::UnstableFeatures;
use rustc_hashes::Hash64;
use rustc_hir::attrs::CollapseMacroDebuginfo;
use rustc_macros::{BlobDecodable, Encodable};
use rustc_span::edition::Edition;
use rustc_span::{RealFileName, RemapPathScopeComponents, SourceFileHashAlgorithm};
use rustc_target::spec::{
    CodeModel, FramePointer, LinkerFlavorCli, MergeFunctions, OnBrokenPipe, PanicStrategy,
    RelocModel, RelroLevel, SanitizerSet, SplitDebuginfo, StackProtector, SymbolVisibility,
    TargetTuple, TlsModel,
};

use crate::config::*;
use crate::search_paths::SearchPath;
use crate::utils::NativeLib;
use crate::{EarlyDiagCtxt, Session, lint};

macro_rules! insert {
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr) => {
        if $sub_hashes
            .insert(stringify!($opt_name), $opt_expr as &dyn dep_tracking::DepTrackingHash)
            .is_some()
        {
            panic!("duplicate key in CLI DepTrackingHash: {}", stringify!($opt_name))
        }
    };
}

macro_rules! hash_opt {
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr, $_for_crate_hash: ident, [UNTRACKED]) => {{}};
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr, $_for_crate_hash: ident, [TRACKED]) => {{ insert!($opt_name, $opt_expr, $sub_hashes) }};
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr, $for_crate_hash: ident, [TRACKED_NO_CRATE_HASH]) => {{
        if !$for_crate_hash {
            insert!($opt_name, $opt_expr, $sub_hashes)
        }
    }};
    ($opt_name:ident, $opt_expr:expr, $sub_hashes:expr, $_for_crate_hash: ident, [SUBSTRUCT]) => {{}};
}

macro_rules! hash_substruct {
    ($opt_name:ident, $opt_expr:expr, $error_format:expr, $for_crate_hash:expr, $hasher:expr, [UNTRACKED]) => {{}};
    ($opt_name:ident, $opt_expr:expr, $error_format:expr, $for_crate_hash:expr, $hasher:expr, [TRACKED]) => {{}};
    ($opt_name:ident, $opt_expr:expr, $error_format:expr, $for_crate_hash:expr, $hasher:expr, [TRACKED_NO_CRATE_HASH]) => {{}};
    ($opt_name:ident, $opt_expr:expr, $error_format:expr, $for_crate_hash:expr, $hasher:expr, [SUBSTRUCT]) => {
        use crate::config::dep_tracking::DepTrackingHash;
        $opt_expr.dep_tracking_hash($for_crate_hash, $error_format).hash(
            $hasher,
            $error_format,
            $for_crate_hash,
        );
    };
}

/// Extended target modifier info.
/// For example, when external target modifier is '-Zregparm=2':
/// Target modifier enum value + user value ('2') from external crate
/// is converted into description: prefix ('Z'), name ('regparm'), tech value ('Some(2)').
pub struct ExtendedTargetModifierInfo {
    /// Flag prefix (usually, 'C' for codegen flags or 'Z' for unstable flags)
    pub prefix: String,
    /// Flag name
    pub name: String,
    /// Flag parsed technical value
    pub tech_value: String,
}

/// A recorded -Zopt_name=opt_value (or -Copt_name=opt_value)
/// which alter the ABI or effectiveness of exploit mitigations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Encodable, BlobDecodable)]
pub struct TargetModifier {
    /// Option enum value
    pub opt: OptionsTargetModifiers,
    /// User-provided option value (before parsing)
    pub value_name: String,
}

mod target_modifier_consistency_check {
    use super::*;
    pub(super) fn sanitizer(l: &TargetModifier, r: Option<&TargetModifier>) -> bool {
        let mut lparsed: SanitizerSet = Default::default();
        let lval = if l.value_name.is_empty() { None } else { Some(l.value_name.as_str()) };
        parse::parse_sanitizers(&mut lparsed, lval);

        let mut rparsed: SanitizerSet = Default::default();
        let rval = r.filter(|v| !v.value_name.is_empty()).map(|v| v.value_name.as_str());
        parse::parse_sanitizers(&mut rparsed, rval);

        // Some sanitizers need to be target modifiers, and some do not.
        // For now, we should mark all sanitizers as target modifiers except for these:
        // AddressSanitizer, LeakSanitizer
        let tmod_sanitizers = SanitizerSet::MEMORY
            | SanitizerSet::THREAD
            | SanitizerSet::HWADDRESS
            | SanitizerSet::CFI
            | SanitizerSet::MEMTAG
            | SanitizerSet::SHADOWCALLSTACK
            | SanitizerSet::KCFI
            | SanitizerSet::KERNELADDRESS
            | SanitizerSet::SAFESTACK
            | SanitizerSet::DATAFLOW;

        lparsed & tmod_sanitizers == rparsed & tmod_sanitizers
    }
    pub(super) fn sanitizer_cfi_normalize_integers(
        sess: &Session,
        l: &TargetModifier,
        r: Option<&TargetModifier>,
    ) -> bool {
        // For kCFI, the helper flag -Zsanitizer-cfi-normalize-integers should also be a target modifier
        if sess.sanitizers().contains(SanitizerSet::KCFI) {
            if let Some(r) = r {
                return l.extend().tech_value == r.extend().tech_value;
            } else {
                return false;
            }
        }
        true
    }
}

impl TargetModifier {
    pub fn extend(&self) -> ExtendedTargetModifierInfo {
        self.opt.reparse(&self.value_name)
    }
    // Custom consistency check for target modifiers (or default `l.tech_value == r.tech_value`)
    // When other is None, consistency with default value is checked
    pub fn consistent(&self, sess: &Session, other: Option<&TargetModifier>) -> bool {
        assert!(other.is_none() || self.opt == other.unwrap().opt);
        match self.opt {
            OptionsTargetModifiers::UnstableOptions(unstable) => match unstable {
                UnstableOptionsTargetModifiers::sanitizer => {
                    return target_modifier_consistency_check::sanitizer(self, other);
                }
                UnstableOptionsTargetModifiers::sanitizer_cfi_normalize_integers => {
                    return target_modifier_consistency_check::sanitizer_cfi_normalize_integers(
                        sess, self, other,
                    );
                }
                _ => {}
            },
            _ => {}
        };
        match other {
            Some(other) => self.extend().tech_value == other.extend().tech_value,
            None => false,
        }
    }
}

fn tmod_push_impl(
    opt: OptionsTargetModifiers,
    tmod_vals: &BTreeMap<OptionsTargetModifiers, String>,
    tmods: &mut Vec<TargetModifier>,
) {
    if let Some(v) = tmod_vals.get(&opt) {
        tmods.push(TargetModifier { opt, value_name: v.clone() })
    }
}

macro_rules! tmod_push {
    ($struct_name:ident, $tmod_enum_name:ident, $opt_name:ident, $opt_expr:expr, $init:expr, $mods:expr, $tmod_vals:expr) => {
        if *$opt_expr != $init {
            tmod_push_impl(
                OptionsTargetModifiers::$struct_name($tmod_enum_name::$opt_name),
                $tmod_vals,
                $mods,
            );
        }
    };
}

macro_rules! gather_tmods {
    ($struct_name:ident, $tmod_enum_name:ident, $opt_name:ident, $opt_expr:expr, $init:expr, $mods:expr, $tmod_vals:expr,
        [SUBSTRUCT], [TARGET_MODIFIER]) => {
        compile_error!("SUBSTRUCT can't be target modifier");
    };
    ($struct_name:ident, $tmod_enum_name:ident, $opt_name:ident, $opt_expr:expr, $init:expr, $mods:expr, $tmod_vals:expr,
        [UNTRACKED], [TARGET_MODIFIER]) => {
        tmod_push!($struct_name, $tmod_enum_name, $opt_name, $opt_expr, $init, $mods, $tmod_vals)
    };
    ($struct_name:ident, $tmod_enum_name:ident, $opt_name:ident, $opt_expr:expr, $init:expr, $mods:expr, $tmod_vals:expr,
        [TRACKED], [TARGET_MODIFIER]) => {
        tmod_push!($struct_name, $tmod_enum_name, $opt_name, $opt_expr, $init, $mods, $tmod_vals)
    };
    ($struct_name:ident, $tmod_enum_name:ident, $opt_name:ident, $opt_expr:expr, $init:expr, $mods:expr, $tmod_vals:expr,
        [TRACKED_NO_CRATE_HASH], [TARGET_MODIFIER]) => {
        tmod_push!($struct_name, $tmod_enum_name, $opt_name, $opt_expr, $init, $mods, $tmod_vals)
    };
    ($struct_name:ident, $tmod_enum_name:ident, $opt_name:ident, $opt_expr:expr, $init:expr, $mods:expr, $tmod_vals:expr,
        [SUBSTRUCT], []) => {
        $opt_expr.gather_target_modifiers($mods, $tmod_vals);
    };
    ($struct_name:ident, $tmod_enum_name:ident, $opt_name:ident, $opt_expr:expr, $init:expr, $mods:expr, $tmod_vals:expr,
        [UNTRACKED], []) => {{}};
    ($struct_name:ident, $tmod_enum_name:ident, $opt_name:ident, $opt_expr:expr, $init:expr, $mods:expr, $tmod_vals:expr,
        [TRACKED], []) => {{}};
    ($struct_name:ident, $tmod_enum_name:ident, $opt_name:ident, $opt_expr:expr, $init:expr, $mods:expr, $tmod_vals:expr,
        [TRACKED_NO_CRATE_HASH], []) => {{}};
}

macro_rules! gather_tmods_top_level {
    ($_opt_name:ident, $opt_expr:expr, $mods:expr, $tmod_vals:expr, [SUBSTRUCT $substruct_enum:ident]) => {
        $opt_expr.gather_target_modifiers($mods, $tmod_vals);
    };
    ($opt_name:ident, $opt_expr:expr, $mods:expr, $tmod_vals:expr, [$non_substruct:ident TARGET_MODIFIER]) => {
        compile_error!("Top level option can't be target modifier");
    };
    ($opt_name:ident, $opt_expr:expr, $mods:expr, $tmod_vals:expr, [$non_substruct:ident]) => {};
}

/// Macro for generating OptionsTargetsModifiers top-level enum with impl.
/// Will generate something like:
/// ```rust,ignore (illustrative)
/// pub enum OptionsTargetModifiers {
///     CodegenOptions(CodegenOptionsTargetModifiers),
///     UnstableOptions(UnstableOptionsTargetModifiers),
/// }
/// impl OptionsTargetModifiers {
///     pub fn reparse(&self, user_value: &str) -> ExtendedTargetModifierInfo {
///         match self {
///             Self::CodegenOptions(v) => v.reparse(user_value),
///             Self::UnstableOptions(v) => v.reparse(user_value),
///         }
///     }
///     pub fn is_target_modifier(flag_name: &str) -> bool {
///         CodegenOptionsTargetModifiers::is_target_modifier(flag_name) ||
///         UnstableOptionsTargetModifiers::is_target_modifier(flag_name)
///     }
/// }
/// ```
macro_rules! top_level_tmod_enum {
    ($( {$($optinfo:tt)*} ),* $(,)*) => {
        top_level_tmod_enum! { @parse {}, (user_value){}; $($($optinfo)*|)* }
    };
    // Termination
    (
        @parse
        {$($variant:tt($substruct_enum:tt))*},
        ($user_value:ident){$($pout:tt)*};
    ) => {
        #[allow(non_camel_case_types)]
        #[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Copy, Clone, Encodable, BlobDecodable)]
        pub enum OptionsTargetModifiers {
            $($variant($substruct_enum)),*
        }
        impl OptionsTargetModifiers {
            #[allow(unused_variables)]
            pub fn reparse(&self, $user_value: &str) -> ExtendedTargetModifierInfo {
                #[allow(unreachable_patterns)]
                match self {
                    $($pout)*
                    _ => panic!("unknown target modifier option: {:?}", *self)
                }
            }
            pub fn is_target_modifier(flag_name: &str) -> bool {
                $($substruct_enum::is_target_modifier(flag_name))||*
            }
        }
    };
    // Adding SUBSTRUCT option group into $eout
    (
        @parse {$($eout:tt)*}, ($puser_value:ident){$($pout:tt)*};
            [SUBSTRUCT $substruct_enum:ident $variant:ident] |
        $($tail:tt)*
    ) => {
        top_level_tmod_enum! {
            @parse
            {
                $($eout)*
                $variant($substruct_enum)
            },
            ($puser_value){
                $($pout)*
                Self::$variant(v) => v.reparse($puser_value),
            };
            $($tail)*
        }
    };
    // Skipping non-target-modifier and non-substruct
    (
        @parse {$($eout:tt)*}, ($puser_value:ident){$($pout:tt)*};
            [$non_substruct:ident] |
        $($tail:tt)*
    ) => {
        top_level_tmod_enum! {
            @parse
            {
                $($eout)*
            },
            ($puser_value){
                $($pout)*
            };
            $($tail)*
        }
    };
}

macro_rules! top_level_options {
    ( $( #[$top_level_attr:meta] )* pub struct Options { $(
        $( #[$attr:meta] )*
        $opt:ident : $t:ty [$dep_tracking_marker:ident $( $tmod:ident $variant:ident )?],
    )* } ) => (
        top_level_tmod_enum!( {$([$dep_tracking_marker $($tmod $variant),*])|*} );

        #[derive(Clone)]
        $( #[$top_level_attr] )*
        pub struct Options {
            $(
                $( #[$attr] )*
                pub $opt: $t
            ),*,
            pub target_modifiers: BTreeMap<OptionsTargetModifiers, String>,
        }

        impl Options {
            pub fn dep_tracking_hash(&self, for_crate_hash: bool) -> Hash64 {
                let mut sub_hashes = BTreeMap::new();
                $({
                    hash_opt!($opt,
                                &self.$opt,
                                &mut sub_hashes,
                                for_crate_hash,
                                [$dep_tracking_marker]);
                })*
                let mut hasher = StableHasher::new();
                dep_tracking::stable_hash(sub_hashes,
                                          &mut hasher,
                                          self.error_format,
                                          for_crate_hash);
                $({
                    hash_substruct!($opt,
                        &self.$opt,
                        self.error_format,
                        for_crate_hash,
                        &mut hasher,
                        [$dep_tracking_marker]);
                })*
                hasher.finish()
            }

            pub fn gather_target_modifiers(&self) -> Vec<TargetModifier> {
                let mut mods = Vec::<TargetModifier>::new();
                $({
                    gather_tmods_top_level!($opt,
                        &self.$opt, &mut mods, &self.target_modifiers,
                        [$dep_tracking_marker $($tmod),*]);
                })*
                mods.sort_by(|a, b| a.opt.cmp(&b.opt));
                mods
            }
        }
    );
}

top_level_options!(
    /// The top-level command-line options struct.
    ///
    /// For each option, one has to specify how it behaves with regard to the
    /// dependency tracking system of incremental compilation. This is done via the
    /// square-bracketed directive after the field type. The options are:
    ///
    /// - `[TRACKED]`
    /// A change in the given field will cause the compiler to completely clear the
    /// incremental compilation cache before proceeding.
    ///
    /// - `[TRACKED_NO_CRATE_HASH]`
    /// Same as `[TRACKED]`, but will not affect the crate hash. This is useful for options that
    /// only affect the incremental cache.
    ///
    /// - `[UNTRACKED]`
    /// Incremental compilation is not influenced by this option.
    ///
    /// - `[SUBSTRUCT]`
    /// Second-level sub-structs containing more options.
    ///
    /// If you add a new option to this struct or one of the sub-structs like
    /// `CodegenOptions`, think about how it influences incremental compilation. If in
    /// doubt, specify `[TRACKED]`, which is always "correct" but might lead to
    /// unnecessary re-compilation.
    #[rustc_lint_opt_ty]
    pub struct Options {
        /// The crate config requested for the session, which may be combined
        /// with additional crate configurations during the compile process.
        #[rustc_lint_opt_deny_field_access("use `TyCtxt::crate_types` instead of this field")]
        crate_types: Vec<CrateType> [TRACKED],
        optimize: OptLevel [TRACKED],
        /// Include the `debug_assertions` flag in dependency tracking, since it
        /// can influence whether overflow checks are done or not.
        debug_assertions: bool [TRACKED],
        debuginfo: DebugInfo [TRACKED],
        lint_opts: Vec<(String, lint::Level)> [TRACKED_NO_CRATE_HASH],
        lint_cap: Option<lint::Level> [TRACKED_NO_CRATE_HASH],
        describe_lints: bool [UNTRACKED],
        output_types: OutputTypes [TRACKED],
        search_paths: Vec<SearchPath> [UNTRACKED],
        libs: Vec<NativeLib> [TRACKED],
        sysroot: Sysroot [UNTRACKED],

        target_triple: TargetTuple [TRACKED],

        /// Effective logical environment used by `env!`/`option_env!` macros
        logical_env: FxIndexMap<String, String> [TRACKED],

        test: bool [TRACKED],
        error_format: ErrorOutputType [UNTRACKED],
        diagnostic_width: Option<usize> [UNTRACKED],

        /// If `Some`, enable incremental compilation, using the given
        /// directory to store intermediate results.
        incremental: Option<PathBuf> [UNTRACKED],
        assert_incr_state: Option<IncrementalStateAssertion> [UNTRACKED],
        /// Set based on the result of the `Config::track_state` callback
        /// for custom drivers to invalidate the incremental cache.
        #[rustc_lint_opt_deny_field_access("should only be used via `Config::track_state`")]
        untracked_state_hash: Hash64 [TRACKED_NO_CRATE_HASH],

        unstable_opts: UnstableOptions [SUBSTRUCT UnstableOptionsTargetModifiers UnstableOptions],
        prints: Vec<PrintRequest> [UNTRACKED],
        cg: CodegenOptions [SUBSTRUCT CodegenOptionsTargetModifiers CodegenOptions],
        externs: Externs [UNTRACKED],
        crate_name: Option<String> [TRACKED],
        /// Indicates how the compiler should treat unstable features.
        unstable_features: UnstableFeatures [TRACKED],

        /// Indicates whether this run of the compiler is actually rustdoc. This
        /// is currently just a hack and will be removed eventually, so please
        /// try to not rely on this too much.
        actually_rustdoc: bool [TRACKED],
        /// Whether name resolver should resolve documentation links.
        resolve_doc_links: ResolveDocLinks [TRACKED],

        /// Control path trimming.
        trimmed_def_paths: bool [TRACKED],

        /// Specifications of codegen units / ThinLTO which are forced as a
        /// result of parsing command line options. These are not necessarily
        /// what rustc was invoked with, but massaged a bit to agree with
        /// commands like `--emit llvm-ir` which they're often incompatible with
        /// if we otherwise use the defaults of rustc.
        #[rustc_lint_opt_deny_field_access("use `Session::codegen_units` instead of this field")]
        cli_forced_codegen_units: Option<usize> [UNTRACKED],
        #[rustc_lint_opt_deny_field_access("use `Session::lto` instead of this field")]
        cli_forced_local_thinlto_off: bool [UNTRACKED],

        /// Remap source path prefixes in all output (messages, object files, debug, etc.).
        remap_path_prefix: Vec<(PathBuf, PathBuf)> [TRACKED_NO_CRATE_HASH],
        /// Defines which scopes of paths should be remapped by `--remap-path-prefix`.
        remap_path_scope: RemapPathScopeComponents [TRACKED_NO_CRATE_HASH],

        /// Base directory containing the `library/` directory for the Rust standard library.
        /// Right now it's always `$sysroot/lib/rustlib/src/rust`
        /// (i.e. the `rustup` `rust-src` component).
        ///
        /// This directory is what the virtual `/rustc/$hash` is translated back to,
        /// if Rust was built with path remapping to `/rustc/$hash` enabled
        /// (the `rust.remap-debuginfo` option in `bootstrap.toml`).
        real_rust_source_base_dir: Option<PathBuf> [TRACKED_NO_CRATE_HASH],

        /// Base directory containing the `compiler/` directory for the rustc sources.
        /// Right now it's always `$sysroot/lib/rustlib/rustc-src/rust`
        /// (i.e. the `rustup` `rustc-dev` component).
        ///
        /// This directory is what the virtual `/rustc-dev/$hash` is translated back to,
        /// if Rust was built with path remapping to `/rustc/$hash` enabled
        /// (the `rust.remap-debuginfo` option in `bootstrap.toml`).
        real_rustc_dev_source_base_dir: Option<PathBuf> [TRACKED_NO_CRATE_HASH],

        edition: Edition [TRACKED],

        /// `true` if we're emitting JSON blobs about each artifact produced
        /// by the compiler.
        json_artifact_notifications: bool [TRACKED],

        /// `true` if we're emitting JSON timings with the start and end of
        /// high-level compilation sections
        json_timings: bool [UNTRACKED],

        /// `true` if we're emitting a JSON blob containing the unused externs
        json_unused_externs: JsonUnusedExterns [UNTRACKED],

        /// `true` if we're emitting a JSON job containing a future-incompat report for lints
        json_future_incompat: bool [TRACKED],

        pretty: Option<PpMode> [UNTRACKED],

        /// The (potentially remapped) working directory
        #[rustc_lint_opt_deny_field_access("use `SourceMap::working_dir` instead of this field")]
        working_dir: RealFileName [TRACKED],

        color: ColorConfig [UNTRACKED],

        verbose: bool [TRACKED_NO_CRATE_HASH],
    }
);

macro_rules! tmod_enum_opt {
    ($struct_name:ident, $tmod_enum_name:ident, $opt:ident, $v:ident) => {
        Some(OptionsTargetModifiers::$struct_name($tmod_enum_name::$opt))
    };
    ($struct_name:ident, $tmod_enum_name:ident, $opt:ident, ) => {
        None
    };
}

macro_rules! tmod_enum {
    ($tmod_enum_name:ident, $prefix:expr, $( {$($optinfo:tt)*} ),* $(,)*) => {
        tmod_enum! { $tmod_enum_name, $prefix, @parse {}, (user_value){}; $($($optinfo)*|)* }
    };
    // Termination
    (
        $tmod_enum_name:ident, $prefix:expr,
        @parse
        {$($eout:tt)*},
        ($user_value:ident){$($pout:tt)*};
    ) => {
        #[allow(non_camel_case_types)]
        #[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Copy, Clone, Encodable, BlobDecodable)]
        pub enum $tmod_enum_name {
            $($eout),*
        }
        impl $tmod_enum_name {
            #[allow(unused_variables)]
            pub fn reparse(&self, $user_value: &str) -> ExtendedTargetModifierInfo {
                #[allow(unreachable_patterns)]
                match self {
                    $($pout)*
                    _ => panic!("unknown target modifier option: {:?}", *self)
                }
            }
            pub fn is_target_modifier(flag_name: &str) -> bool {
                match flag_name.replace('-', "_").as_str() {
                    $(stringify!($eout) => true,)*
                    _ => false,
                }
            }
        }
    };
    // Adding target-modifier option into $eout
    (
        $tmod_enum_name:ident, $prefix:expr,
        @parse {$($eout:tt)*}, ($puser_value:ident){$($pout:tt)*};
            $opt:ident, $parse:ident, $t:ty, [TARGET_MODIFIER] |
        $($tail:tt)*
    ) => {
        tmod_enum! {
            $tmod_enum_name, $prefix,
            @parse
            {
                $($eout)*
                $opt
            },
            ($puser_value){
                $($pout)*
                Self::$opt => {
                    let mut parsed : $t = Default::default();
                    let val = if $puser_value.is_empty() { None } else { Some($puser_value) };
                    parse::$parse(&mut parsed, val);
                    ExtendedTargetModifierInfo {
                        prefix: $prefix.to_string(),
                        name: stringify!($opt).to_string().replace('_', "-"),
                        tech_value: format!("{:?}", parsed),
                    }
                },
            };
            $($tail)*
        }
    };
    // Skipping non-target-modifier
    (
        $tmod_enum_name:ident, $prefix:expr,
        @parse {$($eout:tt)*}, ($puser_value:ident){$($pout:tt)*};
            $opt:ident, $parse:ident, $t:ty, [] |
        $($tail:tt)*
    ) => {
        tmod_enum! {
            $tmod_enum_name, $prefix,
            @parse
            {
                $($eout)*
            },
            ($puser_value){
                $($pout)*
            };
            $($tail)*
        }
    };
}

/// Defines all `CodegenOptions`/`DebuggingOptions` fields and parsers all at once. The goal of this
/// macro is to define an interface that can be programmatically used by the option parser
/// to initialize the struct without hardcoding field names all over the place.
///
/// The goal is to invoke this macro once with the correct fields, and then this macro generates all
/// necessary code. The main gotcha of this macro is the `cgsetters` module which is a bunch of
/// generated code to parse an option into its respective field in the struct. There are a few
/// hand-written parsers for parsing specific types of values in this module.
///
/// Note: this macro's invocation is also parsed by a `syn`-based parser in
/// `src/tools/unstable-book-gen/src/main.rs` to extract unstable option names and descriptions.
/// If the format of this macro changes, that parser may need to be updated as well.
macro_rules! options {
    ($struct_name:ident, $tmod_enum_name:ident, $stat:ident, $optmod:ident, $prefix:expr, $outputname:expr,
     $($( #[$attr:meta] )* $opt:ident : $t:ty = (
        $init:expr,
        $parse:ident,
        [$dep_tracking_marker:ident $( $tmod:ident )?],
        $desc:expr
        $(, removed: $removed:ident )?)
     ),* ,) =>
(
    #[derive(Clone)]
    #[rustc_lint_opt_ty]
    pub struct $struct_name { $( $( #[$attr] )* pub $opt: $t),* }

    tmod_enum!( $tmod_enum_name, $prefix, {$($opt, $parse, $t, [$($tmod),*])|*} );

    impl Default for $struct_name {
        fn default() -> $struct_name {
            $struct_name { $($opt: $init),* }
        }
    }

    impl $struct_name {
        pub fn build(
            early_dcx: &EarlyDiagCtxt,
            matches: &getopts::Matches,
            target_modifiers: &mut BTreeMap<OptionsTargetModifiers, String>,
        ) -> $struct_name {
            build_options(early_dcx, matches, target_modifiers, $stat, $prefix, $outputname)
        }

        fn dep_tracking_hash(&self, for_crate_hash: bool, error_format: ErrorOutputType) -> Hash64 {
            let mut sub_hashes = BTreeMap::new();
            $({
                hash_opt!($opt,
                            &self.$opt,
                            &mut sub_hashes,
                            for_crate_hash,
                            [$dep_tracking_marker]);
            })*
            let mut hasher = StableHasher::new();
            dep_tracking::stable_hash(sub_hashes,
                                        &mut hasher,
                                        error_format,
                                        for_crate_hash
                                        );
            hasher.finish()
        }

        pub fn gather_target_modifiers(
            &self,
            _mods: &mut Vec<TargetModifier>,
            _tmod_vals: &BTreeMap<OptionsTargetModifiers, String>,
        ) {
            $({
                gather_tmods!($struct_name, $tmod_enum_name, $opt, &self.$opt, $init, _mods, _tmod_vals,
                    [$dep_tracking_marker], [$($tmod),*]);
            })*
        }
    }

    pub const $stat: OptionDescrs<$struct_name> =
        &[ $( OptionDesc{ name: stringify!($opt), setter: $optmod::$opt,
            type_desc: desc::$parse, desc: $desc, removed: None $( .or(Some(RemovedOption::$removed)) )?,
            tmod: tmod_enum_opt!($struct_name, $tmod_enum_name, $opt, $($tmod),*) } ),* ];

    mod $optmod {
    $(
        pub(super) fn $opt(cg: &mut super::$struct_name, v: Option<&str>) -> bool {
            super::parse::$parse(&mut redirect_field!(cg.$opt), v)
        }
    )*
    }

) }

impl CodegenOptions {
    // JUSTIFICATION: defn of the suggested wrapper fn
    #[allow(rustc::bad_opt_access)]
    pub fn instrument_coverage(&self) -> InstrumentCoverage {
        self.instrument_coverage
    }
}

// Sometimes different options need to build a common structure.
// That structure can be kept in one of the options' fields, the others become dummy.
macro_rules! redirect_field {
    ($cg:ident.link_arg) => {
        $cg.link_args
    };
    ($cg:ident.pre_link_arg) => {
        $cg.pre_link_args
    };
    ($cg:ident.$field:ident) => {
        $cg.$field
    };
}

type OptionSetter<O> = fn(&mut O, v: Option<&str>) -> bool;
type OptionDescrs<O> = &'static [OptionDesc<O>];

/// Indicates whether a removed option should warn or error.
enum RemovedOption {
    Warn,
    Err,
}

pub struct OptionDesc<O> {
    name: &'static str,
    setter: OptionSetter<O>,
    // description for return value/type from mod desc
    type_desc: &'static str,
    // description for option from options table
    desc: &'static str,
    removed: Option<RemovedOption>,
    tmod: Option<OptionsTargetModifiers>,
}

impl<O> OptionDesc<O> {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn desc(&self) -> &'static str {
        self.desc
    }
}

fn build_options<O: Default>(
    early_dcx: &EarlyDiagCtxt,
    matches: &getopts::Matches,
    target_modifiers: &mut BTreeMap<OptionsTargetModifiers, String>,
    descrs: OptionDescrs<O>,
    prefix: &str,
    outputname: &str,
) -> O {
    let mut op = O::default();
    for option in matches.opt_strs(prefix) {
        let (key, value) = match option.split_once('=') {
            None => (option, None),
            Some((k, v)) => (k.to_string(), Some(v)),
        };

        let option_to_lookup = key.replace('-', "_");
        match descrs.iter().find(|opt_desc| opt_desc.name == option_to_lookup) {
            Some(OptionDesc { name: _, setter, type_desc, desc, removed, tmod }) => {
                if let Some(removed) = removed {
                    // deprecation works for prefixed options only
                    assert!(!prefix.is_empty());
                    match removed {
                        RemovedOption::Warn => {
                            early_dcx.early_warn(format!("`-{prefix} {key}`: {desc}"))
                        }
                        RemovedOption::Err => {
                            early_dcx.early_fatal(format!("`-{prefix} {key}`: {desc}"))
                        }
                    }
                }
                if !setter(&mut op, value) {
                    match value {
                        None => early_dcx.early_fatal(
                            format!(
                                "{outputname} option `{key}` requires {type_desc} (`-{prefix} {key}=<value>`)"
                            ),
                        ),
                        Some(value) => early_dcx.early_fatal(
                            format!(
                                "incorrect value `{value}` for {outputname} option `{key}` - {type_desc} was expected"
                            ),
                        ),
                    }
                }
                if let Some(tmod) = *tmod {
                    let v = value.map_or(String::new(), ToOwned::to_owned);
                    target_modifiers.insert(tmod, v);
                }
            }
            None => early_dcx.early_fatal(format!("unknown {outputname} option: `{key}`")),
        }
    }
    op
}

#[allow(non_upper_case_globals)]
mod desc {
    pub(crate) const parse_ignore: &str = "<ignored>"; // should not be user-visible
    pub(crate) const parse_no_value: &str = "no value";
    pub(crate) const parse_bool: &str =
        "one of: `y`, `yes`, `on`, `true`, `n`, `no`, `off` or `false`";
    pub(crate) const parse_opt_bool: &str = parse_bool;
    pub(crate) const parse_string: &str = "a string";
    pub(crate) const parse_opt_string: &str = parse_string;
    pub(crate) const parse_string_push: &str = parse_string;
    pub(crate) const parse_opt_pathbuf: &str = "a path";
    pub(crate) const parse_list: &str = "a space-separated list of strings";
    pub(crate) const parse_list_with_polarity: &str =
        "a comma-separated list of strings, with elements beginning with + or -";
    pub(crate) const parse_autodiff: &str = "a comma separated list of settings: `Enable`, `PrintSteps`, `PrintTA`, `PrintTAFn`, `PrintAA`, `PrintPerf`, `PrintModBefore`, `PrintModAfter`, `PrintModFinal`, `PrintPasses`, `NoPostopt`, `LooseTypes`, `Inline`, `NoTT`";
    pub(crate) const parse_offload: &str =
        "a comma separated list of settings: `Host=<Absolute-Path>`, `Device`, `Test`";
    pub(crate) const parse_comma_list: &str = "a comma-separated list of strings";
    pub(crate) const parse_opt_comma_list: &str = parse_comma_list;
    pub(crate) const parse_number: &str = "a number";
    pub(crate) const parse_opt_number: &str = parse_number;
    pub(crate) const parse_frame_pointer: &str = "one of `true`/`yes`/`on`, `false`/`no`/`off`, or (with -Zunstable-options) `non-leaf` or `always`";
    pub(crate) const parse_threads: &str = parse_number;
    pub(crate) const parse_time_passes_format: &str = "`text` (default) or `json`";
    pub(crate) const parse_passes: &str = "a space-separated list of passes, or `all`";
    pub(crate) const parse_panic_strategy: &str = "either `unwind`, `abort`, or `immediate-abort`";
    pub(crate) const parse_on_broken_pipe: &str = "either `kill`, `error`, or `inherit`";
    pub(crate) const parse_patchable_function_entry: &str = "either two comma separated integers (total_nops,prefix_nops), with prefix_nops <= total_nops, or one integer (total_nops)";
    pub(crate) const parse_opt_panic_strategy: &str = parse_panic_strategy;
    pub(crate) const parse_relro_level: &str = "one of: `full`, `partial`, or `off`";
    pub(crate) const parse_sanitizers: &str = "comma separated list of sanitizers: `address`, `cfi`, `dataflow`, `hwaddress`, `kcfi`, `kernel-address`, `leak`, `memory`, `memtag`, `safestack`, `shadow-call-stack`, `thread`, or 'realtime'";
    pub(crate) const parse_sanitizer_memory_track_origins: &str = "0, 1, or 2";
    pub(crate) const parse_cfguard: &str =
        "either a boolean (`yes`, `no`, `on`, `off`, etc), `checks`, or `nochecks`";
    pub(crate) const parse_cfprotection: &str = "`none`|`no`|`n` (default), `branch`, `return`, or `full`|`yes`|`y` (equivalent to `branch` and `return`)";
    pub(crate) const parse_debuginfo: &str = "either an integer (0, 1, 2), `none`, `line-directives-only`, `line-tables-only`, `limited`, or `full`";
    pub(crate) const parse_debuginfo_compression: &str = "one of `none`, `zlib`, or `zstd`";
    pub(crate) const parse_mir_strip_debuginfo: &str =
        "one of `none`, `locals-in-tiny-functions`, or `all-locals`";
    pub(crate) const parse_collapse_macro_debuginfo: &str = "one of `no`, `external`, or `yes`";
    pub(crate) const parse_strip: &str = "either `none`, `debuginfo`, or `symbols`";
    pub(crate) const parse_linker_flavor: &str = ::rustc_target::spec::LinkerFlavorCli::one_of();
    pub(crate) const parse_dump_mono_stats: &str = "`markdown` (default) or `json`";
    pub(crate) const parse_instrument_coverage: &str = parse_bool;
    pub(crate) const parse_coverage_options: &str = "`block` | `branch` | `condition`";
    pub(crate) const parse_instrument_xray: &str = "either a boolean (`yes`, `no`, `on`, `off`, etc), or a comma separated list of settings: `always` or `never` (mutually exclusive), `ignore-loops`, `instruction-threshold=N`, `skip-entry`, `skip-exit`";
    pub(crate) const parse_unpretty: &str = "`string` or `string=string`";
    pub(crate) const parse_treat_err_as_bug: &str = "either no value or a non-negative number";
    pub(crate) const parse_next_solver_config: &str =
        "either `globally` (when used without an argument), `coherence` (default) or `no`";
    pub(crate) const parse_lto: &str =
        "either a boolean (`yes`, `no`, `on`, `off`, etc), `thin`, `fat`, or omitted";
    pub(crate) const parse_linker_plugin_lto: &str =
        "either a boolean (`yes`, `no`, `on`, `off`, etc), or the path to the linker plugin";
    pub(crate) const parse_location_detail: &str = "either `none`, or a comma separated list of location details to track: `file`, `line`, or `column`";
    pub(crate) const parse_fmt_debug: &str = "either `full`, `shallow`, or `none`";
    pub(crate) const parse_switch_with_opt_path: &str =
        "an optional path to the profiling data output directory";
    pub(crate) const parse_merge_functions: &str =
        "one of: `disabled`, `trampolines`, or `aliases`";
    pub(crate) const parse_symbol_mangling_version: &str =
        "one of: `legacy`, `v0` (RFC 2603), or `hashed`";
    pub(crate) const parse_opt_symbol_visibility: &str =
        "one of: `hidden`, `protected`, or `interposable`";
    pub(crate) const parse_cargo_src_file_hash: &str =
        "one of `blake3`, `md5`, `sha1`, or `sha256`";
    pub(crate) const parse_src_file_hash: &str = "one of `md5`, `sha1`, or `sha256`";
    pub(crate) const parse_relocation_model: &str =
        "one of supported relocation models (`rustc --print relocation-models`)";
    pub(crate) const parse_code_model: &str =
        "one of supported code models (`rustc --print code-models`)";
    pub(crate) const parse_tls_model: &str =
        "one of supported TLS models (`rustc --print tls-models`)";
    pub(crate) const parse_target_feature: &str = parse_string;
    pub(crate) const parse_terminal_url: &str =
        "either a boolean (`yes`, `no`, `on`, `off`, etc), or `auto`";
    pub(crate) const parse_wasi_exec_model: &str = "either `command` or `reactor`";
    pub(crate) const parse_split_debuginfo: &str =
        "one of supported split-debuginfo modes (`off`, `packed`, or `unpacked`)";
    pub(crate) const parse_split_dwarf_kind: &str =
        "one of supported split dwarf modes (`split` or `single`)";
    pub(crate) const parse_link_self_contained: &str = "one of: `y`, `yes`, `on`, `n`, `no`, `off`, or a list of enabled (`+` prefix) and disabled (`-` prefix) \
        components: `crto`, `libc`, `unwind`, `linker`, `sanitizers`, `mingw`";
    pub(crate) const parse_linker_features: &str =
        "a list of enabled (`+` prefix) and disabled (`-` prefix) features: `lld`";
    pub(crate) const parse_polonius: &str = "either no value or `legacy` (the default), or `next`";
    pub(crate) const parse_annotate_moves: &str =
        "either a boolean (`yes`, `no`, `on`, `off`, etc.), or a size limit in bytes";
    pub(crate) const parse_stack_protector: &str =
        "one of (`none` (default), `basic`, `strong`, or `all`)";
    pub(crate) const parse_branch_protection: &str = "a `,` separated combination of `bti`, `gcs`, `pac-ret`, (optionally with `pc`, `b-key`, `leaf` if `pac-ret` is set)";
    pub(crate) const parse_proc_macro_execution_strategy: &str =
        "one of supported execution strategies (`same-thread`, or `cross-thread`)";
    pub(crate) const parse_inlining_threshold: &str =
        "either a boolean (`yes`, `no`, `on`, `off`, etc), or a non-negative number";
    pub(crate) const parse_llvm_module_flag: &str = "<key>:<type>:<value>:<behavior>. Type must currently be `u32`. Behavior should be one of (`error`, `warning`, `require`, `override`, `append`, `appendunique`, `max`, `min`)";
    pub(crate) const parse_function_return: &str = "`keep` or `thunk-extern`";
    pub(crate) const parse_wasm_c_abi: &str = "`spec`";
    pub(crate) const parse_mir_include_spans: &str =
        "either a boolean (`yes`, `no`, `on`, `off`, etc), or `nll` (default: `nll`)";
    pub(crate) const parse_align: &str = "a number that is a power of 2 between 1 and 2^29";
}

pub mod parse {
    use std::str::FromStr;

    pub(crate) use super::*;
    pub(crate) const MAX_THREADS_CAP: usize = 256;

    /// Ignore the value. Used for removed options where we don't actually want to store
    /// anything in the session.
    pub(crate) fn parse_ignore(_slot: &mut (), _v: Option<&str>) -> bool {
        true
    }

    /// This is for boolean options that don't take a value, and are true simply
    /// by existing on the command-line.
    ///
    /// This style of option is deprecated, and is mainly used by old options
    /// beginning with `no-`.
    pub(crate) fn parse_no_value(slot: &mut bool, v: Option<&str>) -> bool {
        match v {
            None => {
                *slot = true;
                true
            }
            // Trying to specify a value is always forbidden.
            Some(_) => false,
        }
    }

    /// Use this for any boolean option that has a static default.
    pub(crate) fn parse_bool(slot: &mut bool, v: Option<&str>) -> bool {
        match v {
            Some("y") | Some("yes") | Some("on") | Some("true") | None => {
                *slot = true;
                true
            }
            Some("n") | Some("no") | Some("off") | Some("false") => {
                *slot = false;
                true
            }
            _ => false,
        }
    }

    /// Use this for any boolean option that lacks a static default. (The
    /// actions taken when such an option is not specified will depend on
    /// other factors, such as other options, or target options.)
    pub(crate) fn parse_opt_bool(slot: &mut Option<bool>, v: Option<&str>) -> bool {
        match v {
            Some("y") | Some("yes") | Some("on") | Some("true") | None => {
                *slot = Some(true);
                true
            }
            Some("n") | Some("no") | Some("off") | Some("false") => {
                *slot = Some(false);
                true
            }
            _ => false,
        }
    }

    /// Parses whether polonius is enabled, and if so, which version.
    pub(crate) fn parse_polonius(slot: &mut Polonius, v: Option<&str>) -> bool {
        match v {
            Some("legacy") | None => {
                *slot = Polonius::Legacy;
                true
            }
            Some("next") => {
                *slot = Polonius::Next;
                true
            }
            _ => false,
        }
    }

    pub(crate) fn parse_annotate_moves(slot: &mut AnnotateMoves, v: Option<&str>) -> bool {
        let mut bslot = false;
        let mut nslot = 0u64;

        *slot = match v {
            // No value provided: -Z annotate-moves (enable with default limit)
            None => AnnotateMoves::Enabled(None),
            // Explicit boolean value provided: -Z annotate-moves=yes/no
            s @ Some(_) if parse_bool(&mut bslot, s) => {
                if bslot {
                    AnnotateMoves::Enabled(None)
                } else {
                    AnnotateMoves::Disabled
                }
            }
            // With numeric limit provided: -Z annotate-moves=1234
            s @ Some(_) if parse_number(&mut nslot, s) => AnnotateMoves::Enabled(Some(nslot)),
            _ => return false,
        };

        true
    }

    /// Use this for any string option that has a static default.
    pub(crate) fn parse_string(slot: &mut String, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                *slot = s.to_string();
                true
            }
            None => false,
        }
    }

    /// Use this for any string option that lacks a static default.
    pub(crate) fn parse_opt_string(slot: &mut Option<String>, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                *slot = Some(s.to_string());
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_opt_pathbuf(slot: &mut Option<PathBuf>, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                *slot = Some(PathBuf::from(s));
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_string_push(slot: &mut Vec<String>, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                slot.push(s.to_string());
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_list(slot: &mut Vec<String>, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                slot.extend(s.split_whitespace().map(|s| s.to_string()));
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_list_with_polarity(
        slot: &mut Vec<(String, bool)>,
        v: Option<&str>,
    ) -> bool {
        match v {
            Some(s) => {
                for s in s.split(',') {
                    let Some(pass_name) = s.strip_prefix(&['+', '-'][..]) else { return false };
                    slot.push((pass_name.to_string(), &s[..1] == "+"));
                }
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_fmt_debug(opt: &mut FmtDebug, v: Option<&str>) -> bool {
        *opt = match v {
            Some("full") => FmtDebug::Full,
            Some("shallow") => FmtDebug::Shallow,
            Some("none") => FmtDebug::None,
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_location_detail(ld: &mut LocationDetail, v: Option<&str>) -> bool {
        if let Some(v) = v {
            ld.line = false;
            ld.file = false;
            ld.column = false;
            if v == "none" {
                return true;
            }
            for s in v.split(',') {
                match s {
                    "file" => ld.file = true,
                    "line" => ld.line = true,
                    "column" => ld.column = true,
                    _ => return false,
                }
            }
            true
        } else {
            false
        }
    }

    pub(crate) fn parse_comma_list(slot: &mut Vec<String>, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                let mut v: Vec<_> = s.split(',').map(|s| s.to_string()).collect();
                v.sort_unstable();
                *slot = v;
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_opt_comma_list(slot: &mut Option<Vec<String>>, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                let mut v: Vec<_> = s.split(',').map(|s| s.to_string()).collect();
                v.sort_unstable();
                *slot = Some(v);
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_threads(slot: &mut usize, v: Option<&str>) -> bool {
        let ret = match v.and_then(|s| s.parse().ok()) {
            Some(0) => {
                *slot = std::thread::available_parallelism().map_or(1, NonZero::<usize>::get);
                true
            }
            Some(i) => {
                *slot = i;
                true
            }
            None => false,
        };
        // We want to cap the number of threads here to avoid large numbers like 999999 and compiler panics.
        // This solution was suggested here https://github.com/rust-lang/rust/issues/117638#issuecomment-1800925067
        *slot = slot.clone().min(MAX_THREADS_CAP);
        ret
    }

    /// Use this for any numeric option that has a static default.
    pub(crate) fn parse_number<T: Copy + FromStr>(slot: &mut T, v: Option<&str>) -> bool {
        match v.and_then(|s| s.parse().ok()) {
            Some(i) => {
                *slot = i;
                true
            }
            None => false,
        }
    }

    /// Use this for any numeric option that lacks a static default.
    pub(crate) fn parse_opt_number<T: Copy + FromStr>(
        slot: &mut Option<T>,
        v: Option<&str>,
    ) -> bool {
        match v {
            Some(s) => {
                *slot = s.parse().ok();
                slot.is_some()
            }
            None => false,
        }
    }

    pub(crate) fn parse_frame_pointer(slot: &mut FramePointer, v: Option<&str>) -> bool {
        let mut yes = false;
        match v {
            _ if parse_bool(&mut yes, v) && yes => slot.ratchet(FramePointer::Always),
            _ if parse_bool(&mut yes, v) => slot.ratchet(FramePointer::MayOmit),
            Some("always") => slot.ratchet(FramePointer::Always),
            Some("non-leaf") => slot.ratchet(FramePointer::NonLeaf),
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_passes(slot: &mut Passes, v: Option<&str>) -> bool {
        match v {
            Some("all") => {
                *slot = Passes::All;
                true
            }
            v => {
                let mut passes = vec![];
                if parse_list(&mut passes, v) {
                    slot.extend(passes);
                    true
                } else {
                    false
                }
            }
        }
    }

    pub(crate) fn parse_opt_panic_strategy(
        slot: &mut Option<PanicStrategy>,
        v: Option<&str>,
    ) -> bool {
        match v {
            Some("unwind") => *slot = Some(PanicStrategy::Unwind),
            Some("abort") => *slot = Some(PanicStrategy::Abort),
            Some("immediate-abort") => *slot = Some(PanicStrategy::ImmediateAbort),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_panic_strategy(slot: &mut PanicStrategy, v: Option<&str>) -> bool {
        match v {
            Some("unwind") => *slot = PanicStrategy::Unwind,
            Some("abort") => *slot = PanicStrategy::Abort,
            Some("immediate-abort") => *slot = PanicStrategy::ImmediateAbort,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_on_broken_pipe(slot: &mut OnBrokenPipe, v: Option<&str>) -> bool {
        match v {
            // OnBrokenPipe::Default can't be explicitly specified
            Some("kill") => *slot = OnBrokenPipe::Kill,
            Some("error") => *slot = OnBrokenPipe::Error,
            Some("inherit") => *slot = OnBrokenPipe::Inherit,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_patchable_function_entry(
        slot: &mut PatchableFunctionEntry,
        v: Option<&str>,
    ) -> bool {
        let mut total_nops = 0;
        let mut prefix_nops = 0;

        if !parse_number(&mut total_nops, v) {
            let parts = v.and_then(|v| v.split_once(',')).unzip();
            if !parse_number(&mut total_nops, parts.0) {
                return false;
            }
            if !parse_number(&mut prefix_nops, parts.1) {
                return false;
            }
        }

        if let Some(pfe) =
            PatchableFunctionEntry::from_total_and_prefix_nops(total_nops, prefix_nops)
        {
            *slot = pfe;
            return true;
        }
        false
    }

    pub(crate) fn parse_relro_level(slot: &mut Option<RelroLevel>, v: Option<&str>) -> bool {
        match v {
            Some(s) => match s.parse::<RelroLevel>() {
                Ok(level) => *slot = Some(level),
                _ => return false,
            },
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_sanitizers(slot: &mut SanitizerSet, v: Option<&str>) -> bool {
        if let Some(v) = v {
            for s in v.split(',') {
                *slot |= match s {
                    "address" => SanitizerSet::ADDRESS,
                    "cfi" => SanitizerSet::CFI,
                    "dataflow" => SanitizerSet::DATAFLOW,
                    "kcfi" => SanitizerSet::KCFI,
                    "kernel-address" => SanitizerSet::KERNELADDRESS,
                    "leak" => SanitizerSet::LEAK,
                    "memory" => SanitizerSet::MEMORY,
                    "memtag" => SanitizerSet::MEMTAG,
                    "shadow-call-stack" => SanitizerSet::SHADOWCALLSTACK,
                    "thread" => SanitizerSet::THREAD,
                    "hwaddress" => SanitizerSet::HWADDRESS,
                    "safestack" => SanitizerSet::SAFESTACK,
                    "realtime" => SanitizerSet::REALTIME,
                    _ => return false,
                }
            }
            true
        } else {
            false
        }
    }

    pub(crate) fn parse_sanitizer_memory_track_origins(slot: &mut usize, v: Option<&str>) -> bool {
        match v {
            Some("2") | None => {
                *slot = 2;
                true
            }
            Some("1") => {
                *slot = 1;
                true
            }
            Some("0") => {
                *slot = 0;
                true
            }
            Some(_) => false,
        }
    }

    pub(crate) fn parse_strip(slot: &mut Strip, v: Option<&str>) -> bool {
        match v {
            Some("none") => *slot = Strip::None,
            Some("debuginfo") => *slot = Strip::Debuginfo,
            Some("symbols") => *slot = Strip::Symbols,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_cfguard(slot: &mut CFGuard, v: Option<&str>) -> bool {
        if v.is_some() {
            let mut bool_arg = None;
            if parse_opt_bool(&mut bool_arg, v) {
                *slot = if bool_arg.unwrap() { CFGuard::Checks } else { CFGuard::Disabled };
                return true;
            }
        }

        *slot = match v {
            None => CFGuard::Checks,
            Some("checks") => CFGuard::Checks,
            Some("nochecks") => CFGuard::NoChecks,
            Some(_) => return false,
        };
        true
    }

    pub(crate) fn parse_cfprotection(slot: &mut CFProtection, v: Option<&str>) -> bool {
        if v.is_some() {
            let mut bool_arg = None;
            if parse_opt_bool(&mut bool_arg, v) {
                *slot = if bool_arg.unwrap() { CFProtection::Full } else { CFProtection::None };
                return true;
            }
        }

        *slot = match v {
            None | Some("none") => CFProtection::None,
            Some("branch") => CFProtection::Branch,
            Some("return") => CFProtection::Return,
            Some("full") => CFProtection::Full,
            Some(_) => return false,
        };
        true
    }

    pub(crate) fn parse_debuginfo(slot: &mut DebugInfo, v: Option<&str>) -> bool {
        match v {
            Some("0") | Some("none") => *slot = DebugInfo::None,
            Some("line-directives-only") => *slot = DebugInfo::LineDirectivesOnly,
            Some("line-tables-only") => *slot = DebugInfo::LineTablesOnly,
            Some("1") | Some("limited") => *slot = DebugInfo::Limited,
            Some("2") | Some("full") => *slot = DebugInfo::Full,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_debuginfo_compression(
        slot: &mut DebugInfoCompression,
        v: Option<&str>,
    ) -> bool {
        match v {
            Some("none") => *slot = DebugInfoCompression::None,
            Some("zlib") => *slot = DebugInfoCompression::Zlib,
            Some("zstd") => *slot = DebugInfoCompression::Zstd,
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_mir_strip_debuginfo(slot: &mut MirStripDebugInfo, v: Option<&str>) -> bool {
        match v {
            Some("none") => *slot = MirStripDebugInfo::None,
            Some("locals-in-tiny-functions") => *slot = MirStripDebugInfo::LocalsInTinyFunctions,
            Some("all-locals") => *slot = MirStripDebugInfo::AllLocals,
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_linker_flavor(slot: &mut Option<LinkerFlavorCli>, v: Option<&str>) -> bool {
        match v.and_then(|v| LinkerFlavorCli::from_str(v).ok()) {
            Some(lf) => *slot = Some(lf),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_opt_symbol_visibility(
        slot: &mut Option<SymbolVisibility>,
        v: Option<&str>,
    ) -> bool {
        if let Some(v) = v {
            if let Ok(vis) = SymbolVisibility::from_str(v) {
                *slot = Some(vis);
            } else {
                return false;
            }
        }
        true
    }

    pub(crate) fn parse_unpretty(slot: &mut Option<String>, v: Option<&str>) -> bool {
        match v {
            None => false,
            Some(s) if s.split('=').count() <= 2 => {
                *slot = Some(s.to_string());
                true
            }
            _ => false,
        }
    }

    pub(crate) fn parse_time_passes_format(slot: &mut TimePassesFormat, v: Option<&str>) -> bool {
        match v {
            None => true,
            Some("json") => {
                *slot = TimePassesFormat::Json;
                true
            }
            Some("text") => {
                *slot = TimePassesFormat::Text;
                true
            }
            Some(_) => false,
        }
    }

    pub(crate) fn parse_dump_mono_stats(slot: &mut DumpMonoStatsFormat, v: Option<&str>) -> bool {
        match v {
            None => true,
            Some("json") => {
                *slot = DumpMonoStatsFormat::Json;
                true
            }
            Some("markdown") => {
                *slot = DumpMonoStatsFormat::Markdown;
                true
            }
            Some(_) => false,
        }
    }

    pub(crate) fn parse_offload(slot: &mut Vec<Offload>, v: Option<&str>) -> bool {
        let Some(v) = v else {
            *slot = vec![];
            return true;
        };
        let mut v: Vec<&str> = v.split(",").collect();
        v.sort_unstable();
        for &val in v.iter() {
            // Split each entry on '=' if it has an argument
            let (key, arg) = match val.split_once('=') {
                Some((k, a)) => (k, Some(a)),
                None => (val, None),
            };

            let variant = match key {
                "Host" => {
                    if let Some(p) = arg {
                        Offload::Host(p.to_string())
                    } else {
                        return false;
                    }
                }
                "Device" => {
                    if let Some(_) = arg {
                        // Device does not accept a value
                        return false;
                    }
                    Offload::Device
                }
                "Test" => {
                    if let Some(_) = arg {
                        // Test does not accept a value
                        return false;
                    }
                    Offload::Test
                }
                _ => {
                    // FIXME(ZuseZ4): print an error saying which value is not recognized
                    return false;
                }
            };
            slot.push(variant);
        }

        true
    }

    pub(crate) fn parse_autodiff(slot: &mut Vec<AutoDiff>, v: Option<&str>) -> bool {
        let Some(v) = v else {
            *slot = vec![];
            return true;
        };
        let mut v: Vec<&str> = v.split(",").collect();
        v.sort_unstable();
        for &val in v.iter() {
            // Split each entry on '=' if it has an argument
            let (key, arg) = match val.split_once('=') {
                Some((k, a)) => (k, Some(a)),
                None => (val, None),
            };

            let variant = match key {
                "Enable" => AutoDiff::Enable,
                "PrintTA" => AutoDiff::PrintTA,
                "PrintTAFn" => {
                    if let Some(fun) = arg {
                        AutoDiff::PrintTAFn(fun.to_string())
                    } else {
                        return false;
                    }
                }
                "PrintAA" => AutoDiff::PrintAA,
                "PrintPerf" => AutoDiff::PrintPerf,
                "PrintSteps" => AutoDiff::PrintSteps,
                "PrintModBefore" => AutoDiff::PrintModBefore,
                "PrintModAfter" => AutoDiff::PrintModAfter,
                "PrintModFinal" => AutoDiff::PrintModFinal,
                "NoPostopt" => AutoDiff::NoPostopt,
                "PrintPasses" => AutoDiff::PrintPasses,
                "LooseTypes" => AutoDiff::LooseTypes,
                "Inline" => AutoDiff::Inline,
                "NoTT" => AutoDiff::NoTT,
                _ => {
                    // FIXME(ZuseZ4): print an error saying which value is not recognized
                    return false;
                }
            };
            slot.push(variant);
        }

        true
    }

    pub(crate) fn parse_instrument_coverage(
        slot: &mut InstrumentCoverage,
        v: Option<&str>,
    ) -> bool {
        if v.is_some() {
            let mut bool_arg = false;
            if parse_bool(&mut bool_arg, v) {
                *slot = if bool_arg { InstrumentCoverage::Yes } else { InstrumentCoverage::No };
                return true;
            }
        }

        let Some(v) = v else {
            *slot = InstrumentCoverage::Yes;
            return true;
        };

        // Parse values that have historically been accepted by stable compilers,
        // even though they're currently just aliases for boolean values.
        *slot = match v {
            "all" => InstrumentCoverage::Yes,
            "0" => InstrumentCoverage::No,
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_coverage_options(slot: &mut CoverageOptions, v: Option<&str>) -> bool {
        let Some(v) = v else { return true };

        for option in v.split(',') {
            match option {
                "block" => slot.level = CoverageLevel::Block,
                "branch" => slot.level = CoverageLevel::Branch,
                "condition" => slot.level = CoverageLevel::Condition,
                "discard-all-spans-in-codegen" => slot.discard_all_spans_in_codegen = true,
                _ => return false,
            }
        }
        true
    }

    pub(crate) fn parse_instrument_xray(
        slot: &mut Option<InstrumentXRay>,
        v: Option<&str>,
    ) -> bool {
        if v.is_some() {
            let mut bool_arg = None;
            if parse_opt_bool(&mut bool_arg, v) {
                *slot = if bool_arg.unwrap() { Some(InstrumentXRay::default()) } else { None };
                return true;
            }
        }

        let options = slot.get_or_insert_default();
        let mut seen_always = false;
        let mut seen_never = false;
        let mut seen_ignore_loops = false;
        let mut seen_instruction_threshold = false;
        let mut seen_skip_entry = false;
        let mut seen_skip_exit = false;
        for option in v.into_iter().flat_map(|v| v.split(',')) {
            match option {
                "always" if !seen_always && !seen_never => {
                    options.always = true;
                    options.never = false;
                    seen_always = true;
                }
                "never" if !seen_never && !seen_always => {
                    options.never = true;
                    options.always = false;
                    seen_never = true;
                }
                "ignore-loops" if !seen_ignore_loops => {
                    options.ignore_loops = true;
                    seen_ignore_loops = true;
                }
                option
                    if option.starts_with("instruction-threshold")
                        && !seen_instruction_threshold =>
                {
                    let Some(("instruction-threshold", n)) = option.split_once('=') else {
                        return false;
                    };
                    match n.parse() {
                        Ok(n) => options.instruction_threshold = Some(n),
                        Err(_) => return false,
                    }
                    seen_instruction_threshold = true;
                }
                "skip-entry" if !seen_skip_entry => {
                    options.skip_entry = true;
                    seen_skip_entry = true;
                }
                "skip-exit" if !seen_skip_exit => {
                    options.skip_exit = true;
                    seen_skip_exit = true;
                }
                _ => return false,
            }
        }
        true
    }

    pub(crate) fn parse_treat_err_as_bug(
        slot: &mut Option<NonZero<usize>>,
        v: Option<&str>,
    ) -> bool {
        match v {
            Some(s) => match s.parse() {
                Ok(val) => {
                    *slot = Some(val);
                    true
                }
                Err(e) => {
                    *slot = None;
                    e.kind() == &IntErrorKind::Zero
                }
            },
            None => {
                *slot = NonZero::new(1);
                true
            }
        }
    }

    pub(crate) fn parse_next_solver_config(slot: &mut NextSolverConfig, v: Option<&str>) -> bool {
        if let Some(config) = v {
            *slot = match config {
                "no" => NextSolverConfig { coherence: false, globally: false },
                "coherence" => NextSolverConfig { coherence: true, globally: false },
                "globally" => NextSolverConfig { coherence: true, globally: true },
                _ => return false,
            };
        } else {
            *slot = NextSolverConfig { coherence: true, globally: true };
        }

        true
    }

    pub(crate) fn parse_lto(slot: &mut LtoCli, v: Option<&str>) -> bool {
        if v.is_some() {
            let mut bool_arg = None;
            if parse_opt_bool(&mut bool_arg, v) {
                *slot = if bool_arg.unwrap() { LtoCli::Yes } else { LtoCli::No };
                return true;
            }
        }

        *slot = match v {
            None => LtoCli::NoParam,
            Some("thin") => LtoCli::Thin,
            Some("fat") => LtoCli::Fat,
            Some(_) => return false,
        };
        true
    }

    pub(crate) fn parse_linker_plugin_lto(slot: &mut LinkerPluginLto, v: Option<&str>) -> bool {
        if v.is_some() {
            let mut bool_arg = None;
            if parse_opt_bool(&mut bool_arg, v) {
                *slot = if bool_arg.unwrap() {
                    LinkerPluginLto::LinkerPluginAuto
                } else {
                    LinkerPluginLto::Disabled
                };
                return true;
            }
        }

        *slot = match v {
            None => LinkerPluginLto::LinkerPluginAuto,
            Some(path) => LinkerPluginLto::LinkerPlugin(PathBuf::from(path)),
        };
        true
    }

    pub(crate) fn parse_switch_with_opt_path(
        slot: &mut SwitchWithOptPath,
        v: Option<&str>,
    ) -> bool {
        *slot = match v {
            None => SwitchWithOptPath::Enabled(None),
            Some(path) => SwitchWithOptPath::Enabled(Some(PathBuf::from(path))),
        };
        true
    }

    pub(crate) fn parse_merge_functions(
        slot: &mut Option<MergeFunctions>,
        v: Option<&str>,
    ) -> bool {
        match v.and_then(|s| MergeFunctions::from_str(s).ok()) {
            Some(mergefunc) => *slot = Some(mergefunc),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_relocation_model(slot: &mut Option<RelocModel>, v: Option<&str>) -> bool {
        match v.and_then(|s| RelocModel::from_str(s).ok()) {
            Some(relocation_model) => *slot = Some(relocation_model),
            None if v == Some("default") => *slot = None,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_code_model(slot: &mut Option<CodeModel>, v: Option<&str>) -> bool {
        match v.and_then(|s| CodeModel::from_str(s).ok()) {
            Some(code_model) => *slot = Some(code_model),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_tls_model(slot: &mut Option<TlsModel>, v: Option<&str>) -> bool {
        match v.and_then(|s| TlsModel::from_str(s).ok()) {
            Some(tls_model) => *slot = Some(tls_model),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_terminal_url(slot: &mut TerminalUrl, v: Option<&str>) -> bool {
        *slot = match v {
            Some("on" | "" | "yes" | "y") | None => TerminalUrl::Yes,
            Some("off" | "no" | "n") => TerminalUrl::No,
            Some("auto") => TerminalUrl::Auto,
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_symbol_mangling_version(
        slot: &mut Option<SymbolManglingVersion>,
        v: Option<&str>,
    ) -> bool {
        *slot = match v {
            Some("legacy") => Some(SymbolManglingVersion::Legacy),
            Some("v0") => Some(SymbolManglingVersion::V0),
            Some("hashed") => Some(SymbolManglingVersion::Hashed),
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_src_file_hash(
        slot: &mut Option<SourceFileHashAlgorithm>,
        v: Option<&str>,
    ) -> bool {
        match v.and_then(|s| SourceFileHashAlgorithm::from_str(s).ok()) {
            Some(hash_kind) => *slot = Some(hash_kind),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_cargo_src_file_hash(
        slot: &mut Option<SourceFileHashAlgorithm>,
        v: Option<&str>,
    ) -> bool {
        match v.and_then(|s| SourceFileHashAlgorithm::from_str(s).ok()) {
            Some(hash_kind) => {
                *slot = Some(hash_kind);
            }
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_target_feature(slot: &mut String, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                if !slot.is_empty() {
                    slot.push(',');
                }
                slot.push_str(s);
                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_link_self_contained(slot: &mut LinkSelfContained, v: Option<&str>) -> bool {
        // Whenever `-C link-self-contained` is passed without a value, it's an opt-in
        // just like `parse_opt_bool`, the historical value of this flag.
        //
        // 1. Parse historical single bool values
        let s = v.unwrap_or("y");
        match s {
            "y" | "yes" | "on" => {
                slot.set_all_explicitly(true);
                return true;
            }
            "n" | "no" | "off" => {
                slot.set_all_explicitly(false);
                return true;
            }
            _ => {}
        }

        // 2. Parse a list of enabled and disabled components.
        for comp in s.split(',') {
            if slot.handle_cli_component(comp).is_none() {
                return false;
            }
        }

        true
    }

    /// Parse a comma-separated list of enabled and disabled linker features.
    pub(crate) fn parse_linker_features(slot: &mut LinkerFeaturesCli, v: Option<&str>) -> bool {
        match v {
            Some(s) => {
                for feature in s.split(',') {
                    if slot.handle_cli_feature(feature).is_none() {
                        return false;
                    }
                }

                true
            }
            None => false,
        }
    }

    pub(crate) fn parse_wasi_exec_model(slot: &mut Option<WasiExecModel>, v: Option<&str>) -> bool {
        match v {
            Some("command") => *slot = Some(WasiExecModel::Command),
            Some("reactor") => *slot = Some(WasiExecModel::Reactor),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_split_debuginfo(
        slot: &mut Option<SplitDebuginfo>,
        v: Option<&str>,
    ) -> bool {
        match v.and_then(|s| SplitDebuginfo::from_str(s).ok()) {
            Some(e) => *slot = Some(e),
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_split_dwarf_kind(slot: &mut SplitDwarfKind, v: Option<&str>) -> bool {
        match v.and_then(|s| SplitDwarfKind::from_str(s).ok()) {
            Some(e) => *slot = e,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_stack_protector(slot: &mut StackProtector, v: Option<&str>) -> bool {
        match v.and_then(|s| StackProtector::from_str(s).ok()) {
            Some(ssp) => *slot = ssp,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_branch_protection(
        slot: &mut Option<BranchProtection>,
        v: Option<&str>,
    ) -> bool {
        match v {
            Some(s) => {
                let slot = slot.get_or_insert_default();
                for opt in s.split(',') {
                    match opt {
                        "bti" => slot.bti = true,
                        "pac-ret" if slot.pac_ret.is_none() => {
                            slot.pac_ret = Some(PacRet { leaf: false, pc: false, key: PAuthKey::A })
                        }
                        "leaf" => match slot.pac_ret.as_mut() {
                            Some(pac) => pac.leaf = true,
                            _ => return false,
                        },
                        "b-key" => match slot.pac_ret.as_mut() {
                            Some(pac) => pac.key = PAuthKey::B,
                            _ => return false,
                        },
                        "pc" => match slot.pac_ret.as_mut() {
                            Some(pac) => pac.pc = true,
                            _ => return false,
                        },
                        "gcs" => slot.gcs = true,
                        _ => return false,
                    };
                }
            }
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_collapse_macro_debuginfo(
        slot: &mut CollapseMacroDebuginfo,
        v: Option<&str>,
    ) -> bool {
        if v.is_some() {
            let mut bool_arg = None;
            if parse_opt_bool(&mut bool_arg, v) {
                *slot = if bool_arg.unwrap() {
                    CollapseMacroDebuginfo::Yes
                } else {
                    CollapseMacroDebuginfo::No
                };
                return true;
            }
        }

        *slot = match v {
            Some("external") => CollapseMacroDebuginfo::External,
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_proc_macro_execution_strategy(
        slot: &mut ProcMacroExecutionStrategy,
        v: Option<&str>,
    ) -> bool {
        *slot = match v {
            Some("same-thread") => ProcMacroExecutionStrategy::SameThread,
            Some("cross-thread") => ProcMacroExecutionStrategy::CrossThread,
            _ => return false,
        };
        true
    }

    pub(crate) fn parse_inlining_threshold(slot: &mut InliningThreshold, v: Option<&str>) -> bool {
        match v {
            Some("always" | "yes") => {
                *slot = InliningThreshold::Always;
            }
            Some("never") => {
                *slot = InliningThreshold::Never;
            }
            Some(v) => {
                if let Ok(threshold) = v.parse() {
                    *slot = InliningThreshold::Sometimes(threshold);
                } else {
                    return false;
                }
            }
            None => return false,
        }
        true
    }

    pub(crate) fn parse_llvm_module_flag(
        slot: &mut Vec<(String, u32, String)>,
        v: Option<&str>,
    ) -> bool {
        let elements = v.unwrap_or_default().split(':').collect::<Vec<_>>();
        let [key, md_type, value, behavior] = elements.as_slice() else {
            return false;
        };
        if *md_type != "u32" {
            // Currently we only support u32 metadata flags, but require the
            // type for forward-compatibility.
            return false;
        }
        let Ok(value) = value.parse::<u32>() else {
            return false;
        };
        let behavior = behavior.to_lowercase();
        let all_behaviors =
            ["error", "warning", "require", "override", "append", "appendunique", "max", "min"];
        if !all_behaviors.contains(&behavior.as_str()) {
            return false;
        }

        slot.push((key.to_string(), value, behavior));
        true
    }

    pub(crate) fn parse_function_return(slot: &mut FunctionReturn, v: Option<&str>) -> bool {
        match v {
            Some("keep") => *slot = FunctionReturn::Keep,
            Some("thunk-extern") => *slot = FunctionReturn::ThunkExtern,
            _ => return false,
        }
        true
    }

    pub(crate) fn parse_wasm_c_abi(_slot: &mut (), v: Option<&str>) -> bool {
        v == Some("spec")
    }

    pub(crate) fn parse_mir_include_spans(slot: &mut MirIncludeSpans, v: Option<&str>) -> bool {
        *slot = match v {
            Some("on" | "yes" | "y" | "true") | None => MirIncludeSpans::On,
            Some("off" | "no" | "n" | "false") => MirIncludeSpans::Off,
            Some("nll") => MirIncludeSpans::Nll,
            _ => return false,
        };

        true
    }

    pub(crate) fn parse_align(slot: &mut Option<Align>, v: Option<&str>) -> bool {
        let mut bytes = 0u64;
        if !parse_number(&mut bytes, v) {
            return false;
        }

        let Ok(align) = Align::from_bytes(bytes) else {
            return false;
        };

        *slot = Some(align);

        true
    }
}

options! {
    CodegenOptions, CodegenOptionsTargetModifiers, CG_OPTIONS, cgopts, "C", "codegen",

    // If you add a new option, please update:
    // - compiler/rustc_interface/src/tests.rs
    // - src/doc/rustc/src/codegen-options/index.md

    // tidy-alphabetical-start
    #[rustc_lint_opt_deny_field_access("documented to do nothing")]
    ar: String = (String::new(), parse_string, [UNTRACKED],
        "this option is deprecated and does nothing",
        removed: Warn),
    #[rustc_lint_opt_deny_field_access("use `Session::code_model` instead of this field")]
    code_model: Option<CodeModel> = (None, parse_code_model, [TRACKED],
        "choose the code model to use (`rustc --print code-models` for details)"),
    codegen_units: Option<usize> = (None, parse_opt_number, [UNTRACKED],
        "divide crate into N units to optimize in parallel"),
    collapse_macro_debuginfo: CollapseMacroDebuginfo = (CollapseMacroDebuginfo::Unspecified,
        parse_collapse_macro_debuginfo, [TRACKED],
        "set option to collapse debuginfo for macros"),
    control_flow_guard: CFGuard = (CFGuard::Disabled, parse_cfguard, [TRACKED],
        "use Windows Control Flow Guard (default: no)"),
    debug_assertions: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "explicitly enable the `cfg(debug_assertions)` directive"),
    debuginfo: DebugInfo = (DebugInfo::None, parse_debuginfo, [TRACKED],
        "debug info emission level (0-2, none, line-directives-only, \
        line-tables-only, limited, or full; default: 0)"),
    default_linker_libraries: bool = (false, parse_bool, [UNTRACKED],
        "allow the linker to link its default libraries (default: no)"),
    dlltool: Option<PathBuf> = (None, parse_opt_pathbuf, [UNTRACKED],
        "import library generation tool (ignored except when targeting windows-gnu)"),
    #[rustc_lint_opt_deny_field_access("use `Session::dwarf_version` instead of this field")]
    dwarf_version: Option<u32> = (None, parse_opt_number, [TRACKED],
        "version of DWARF debug information to emit (default: 2 or 4, depending on platform)"),
    embed_bitcode: bool = (true, parse_bool, [TRACKED],
        "emit bitcode in rlibs (default: yes)"),
    extra_filename: String = (String::new(), parse_string, [UNTRACKED],
        "extra data to put in each output filename"),
    force_frame_pointers: FramePointer = (FramePointer::MayOmit, parse_frame_pointer, [TRACKED],
        "force use of the frame pointers"),
    #[rustc_lint_opt_deny_field_access("use `Session::must_emit_unwind_tables` instead of this field")]
    force_unwind_tables: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "force use of unwind tables"),
    help: bool = (false, parse_no_value, [UNTRACKED], "Print codegen options"),
    incremental: Option<String> = (None, parse_opt_string, [UNTRACKED],
        "enable incremental compilation"),
    #[rustc_lint_opt_deny_field_access("documented to do nothing")]
    inline_threshold: Option<u32> = (None, parse_opt_number, [UNTRACKED],
        "this option is deprecated and does nothing \
        (consider using `-Cllvm-args=--inline-threshold=...`)",
        removed: Warn),
    #[rustc_lint_opt_deny_field_access("use `Session::instrument_coverage` instead of this field")]
    instrument_coverage: InstrumentCoverage = (InstrumentCoverage::No, parse_instrument_coverage, [TRACKED],
        "instrument the generated code to support LLVM source-based code coverage reports \
        (note, the compiler build config must include `profiler = true`); \
        implies `-C symbol-mangling-version=v0`"),
    jump_tables: bool = (true, parse_bool, [TRACKED],
        "allow jump table and lookup table generation from switch case lowering (default: yes)"),
    link_arg: (/* redirected to link_args */) = ((), parse_string_push, [UNTRACKED],
        "a single extra argument to append to the linker invocation (can be used several times)"),
    link_args: Vec<String> = (Vec::new(), parse_list, [UNTRACKED],
        "extra arguments to append to the linker invocation (space separated)"),
    #[rustc_lint_opt_deny_field_access("use `Session::link_dead_code` instead of this field")]
    link_dead_code: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "try to generate and link dead code (default: no)"),
    link_self_contained: LinkSelfContained = (LinkSelfContained::default(), parse_link_self_contained, [UNTRACKED],
        "control whether to link Rust provided C objects/libraries or rely \
        on a C toolchain or linker installed in the system"),
    linker: Option<PathBuf> = (None, parse_opt_pathbuf, [UNTRACKED],
        "system linker to link outputs with"),
    linker_features: LinkerFeaturesCli = (LinkerFeaturesCli::default(), parse_linker_features, [UNTRACKED],
        "a comma-separated list of linker features to enable (+) or disable (-): `lld`"),
    linker_flavor: Option<LinkerFlavorCli> = (None, parse_linker_flavor, [UNTRACKED],
        "linker flavor"),
    linker_plugin_lto: LinkerPluginLto = (LinkerPluginLto::Disabled,
        parse_linker_plugin_lto, [TRACKED],
        "generate build artifacts that are compatible with linker-based LTO"),
    llvm_args: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "a list of arguments to pass to LLVM (space separated)"),
    #[rustc_lint_opt_deny_field_access("use `Session::lto` instead of this field")]
    lto: LtoCli = (LtoCli::Unspecified, parse_lto, [TRACKED],
        "perform LLVM link-time optimizations"),
    metadata: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "metadata to mangle symbol names with"),
    no_prepopulate_passes: bool = (false, parse_no_value, [TRACKED],
        "give an empty list of passes to the pass manager"),
    no_redzone: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "disable the use of the redzone"),
    #[rustc_lint_opt_deny_field_access("documented to do nothing")]
    no_stack_check: bool = (false, parse_no_value, [UNTRACKED],
        "this option is deprecated and does nothing",
        removed: Warn),
    no_vectorize_loops: bool = (false, parse_no_value, [TRACKED],
        "disable loop vectorization optimization passes"),
    no_vectorize_slp: bool = (false, parse_no_value, [TRACKED],
        "disable LLVM's SLP vectorization pass"),
    opt_level: String = ("0".to_string(), parse_string, [TRACKED],
        "optimization level (0-3, s, or z; default: 0)"),
    #[rustc_lint_opt_deny_field_access("use `Session::overflow_checks` instead of this field")]
    overflow_checks: Option<bool> = (None, parse_opt_bool, [TRACKED],
        "use overflow checks for integer arithmetic"),
    #[rustc_lint_opt_deny_field_access("use `Session::panic_strategy` instead of this field")]
    panic: Option<PanicStrategy> = (None, parse_opt_panic_strategy, [TRACKED],
        "panic strategy to compile crate with"),
    passes: Vec<String> = (Vec::new(), parse_list, [TRACKED],
        "a list of extra LLVM passes to run (space separated)"),
    prefer_dynamic: bool = (false, parse_bool, [TRACKED],
        "prefer dynamic linking to static linking (default: no)"),
    profile_generate: SwitchWithOptPath = (SwitchWithOptPath::Disabled,
        parse_switch_with_opt_path, [TRACKED],
        "compile the program with profiling instrumentation"),
    profile_use: Option<PathBuf> = (None, parse_opt_pathbuf, [TRACKED],
        "use the given `.profdata` file for profile-guided optimization"),
    #[rustc_lint_opt_deny_field_access("use `Session::relocation_model` instead of this field")]
    relocation_model: Option<RelocModel> = (None, parse_relocation_model, [TRACKED],
        "control generation of position-independent code (PIC) \
        (`rustc --print relocation-models` for details)"),
    relro_level: Option<RelroLevel> = (None, parse_relro_level, [TRACKED],
        "choose which RELRO level to use"),
    remark: Passes = (Passes::Some(Vec::new()), parse_passes, [UNTRACKED],
        "output remarks for these optimization passes (space separated, or \"all\")"),
    rpath: bool = (false, parse_bool, [UNTRACKED],
        "set rpath values in libs/exes (default: no)"),
    save_temps: bool = (false, parse_bool, [UNTRACKED],
        "save all temporary output files during compilation (default: no)"),
    #[rustc_lint_opt_deny_field_access("documented to do nothing")]
    soft_float: () = ((), parse_ignore, [UNTRACKED],
        "this option has been removed \
        (use a corresponding *eabi target instead)",
        removed: Err),
    #[rustc_lint_opt_deny_field_access("use `Session::split_debuginfo` instead of this field")]
    split_debuginfo: Option<SplitDebuginfo> = (None, parse_split_debuginfo, [TRACKED],
        "how to handle split-debuginfo, a platform-specific option"),
    strip: Strip = (Strip::None, parse_strip, [UNTRACKED],
        "tell the linker which information to strip (`none` (default), `debuginfo` or `symbols`)"),
    symbol_mangling_version: Option<SymbolManglingVersion> = (None,
        parse_symbol_mangling_version, [TRACKED],
        "which mangling version to use for symbol names ('legacy' (default), 'v0', or 'hashed')"),
    target_cpu: Option<String> = (None, parse_opt_string, [TRACKED],
        "select target processor (`rustc --print target-cpus` for details)"),
    target_feature: String = (String::new(), parse_target_feature, [TRACKED],
        "target specific attributes. (`rustc --print target-features` for details). \
        This feature is unsafe."),
    unsafe_allow_abi_mismatch: Vec<String> = (Vec::new(), parse_comma_list, [UNTRACKED],
        "Allow incompatible target modifiers in dependency crates (comma separated list)"),
    // tidy-alphabetical-end

    // If you add a new option, please update:
    // - compiler/rustc_interface/src/tests.rs
    // - src/doc/rustc/src/codegen-options/index.md
}

include!("options/unstable.rs");
