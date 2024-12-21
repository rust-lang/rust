use rustc_abi::Align;
use rustc_attr_parsing::{InlineAttr, InstructionSetAttr, OptimizeAttr};
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use rustc_span::{Symbol, sym};
use rustc_target::spec::SanitizerSet;

use crate::mir::mono::Linkage;
use crate::ty::TyCtxt;

#[derive(Clone, TyEncodable, TyDecodable, HashStable, Debug)]
pub struct CodegenFnAttrs {
    pub flags: CodegenFnAttrFlags,
    /// Parsed representation of the `#[inline]` attribute
    pub inline: InlineAttr,
    /// Parsed representation of the `#[optimize]` attribute
    pub optimize: OptimizeAttr,
    /// The `#[export_name = "..."]` attribute, indicating a custom symbol a
    /// function should be exported under
    pub export_name: Option<Symbol>,
    /// The `#[link_name = "..."]` attribute, indicating a custom symbol an
    /// imported function should be imported as. Note that `export_name`
    /// probably isn't set when this is set, this is for foreign items while
    /// `#[export_name]` is for Rust-defined functions.
    pub link_name: Option<Symbol>,
    /// The `#[link_ordinal = "..."]` attribute, indicating an ordinal an
    /// imported function has in the dynamic library. Note that this must not
    /// be set when `link_name` is set. This is for foreign items with the
    /// "raw-dylib" kind.
    pub link_ordinal: Option<u16>,
    /// The `#[target_feature(enable = "...")]` attribute and the enabled
    /// features (only enabled features are supported right now).
    pub target_features: Vec<TargetFeature>,
    /// The `#[linkage = "..."]` attribute on Rust-defined items and the value we found.
    pub linkage: Option<Linkage>,
    /// The `#[linkage = "..."]` attribute on foreign items and the value we found.
    pub import_linkage: Option<Linkage>,
    /// The `#[link_section = "..."]` attribute, or what executable section this
    /// should be placed in.
    pub link_section: Option<Symbol>,
    /// The `#[no_sanitize(...)]` attribute. Indicates sanitizers for which
    /// instrumentation should be disabled inside the annotated function.
    pub no_sanitize: SanitizerSet,
    /// The `#[instruction_set(set)]` attribute. Indicates if the generated code should
    /// be generated against a specific instruction set. Only usable on architectures which allow
    /// switching between multiple instruction sets.
    pub instruction_set: Option<InstructionSetAttr>,
    /// The `#[repr(align(...))]` attribute. Indicates the value of which the function should be
    /// aligned to.
    pub alignment: Option<Align>,
    /// The `#[patchable_function_entry(...)]` attribute. Indicates how many nops should be around
    /// the function entry.
    pub patchable_function_entry: Option<PatchableFunctionEntry>,
    /// Whether to generate debuginfo that instructs libstd's panic handler to
    /// skip this frame when printing short backtraces.
    pub skip_short_backtrace: Option<SkipShortBacktrace>,
}

#[derive(Copy, Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct TargetFeature {
    /// The name of the target feature (e.g. "avx")
    pub name: Symbol,
    /// The feature is implied by another feature, rather than explicitly added by the
    /// `#[target_feature]` attribute
    pub implied: bool,
}

#[derive(Copy, Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub struct PatchableFunctionEntry {
    /// Nops to prepend to the function
    prefix: u8,
    /// Nops after entry, but before body
    entry: u8,
}

impl PatchableFunctionEntry {
    pub fn from_config(config: rustc_session::config::PatchableFunctionEntry) -> Self {
        Self { prefix: config.prefix(), entry: config.entry() }
    }
    pub fn from_prefix_and_entry(prefix: u8, entry: u8) -> Self {
        Self { prefix, entry }
    }
    pub fn prefix(&self) -> u8 {
        self.prefix
    }
    pub fn entry(&self) -> u8 {
        self.entry
    }
}

// #[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
#[derive(Copy, Clone, Debug, TyEncodable, TyDecodable, HashStable)]
pub enum SkipShortBacktrace {
    ThisFrameOnly,
    Start,
    End,
}

#[derive(Diagnostic)]
#[diag(middle_multiple_short_backtrace_attrs)]
struct MultipleShortBacktraceAttrs {
    #[primary_span]
    #[label]
    pub span: Span,
    #[note]
    start: Option<Span>,
    #[note]
    end: Option<Span>,
    #[note]
    skip: Option<Span>,
}

impl SkipShortBacktrace {
    pub fn lookup(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<SkipShortBacktrace> {
        let skip = tcx.get_attr(def_id, sym::rustc_skip_short_backtrace);
        let start = tcx.get_attr(def_id, sym::rustc_start_short_backtrace);
        let end = tcx.get_attr(def_id, sym::rustc_end_short_backtrace);

        // TODO: need to add a visitor somewhere that errors if this is applied other than to a module or function

        let kind = match (skip.is_some(), start.is_some(), end.is_some()) {
            (true, false, false) => SkipShortBacktrace::ThisFrameOnly,
            (false, true, false) => SkipShortBacktrace::Start,
            (false, false, true) => SkipShortBacktrace::End,
            (false, false, false) => {
                let parent_mod = tcx.parent_module_from_def_id(def_id);
                if tcx.get_attr(parent_mod, sym::rustc_skip_short_backtrace).is_some() {
                    SkipShortBacktrace::ThisFrameOnly
                } else {
                    return None;
                }
            }
            _ => {
                tcx.dcx().emit_err(MultipleShortBacktraceAttrs {
                    span: tcx.def_span(def_id),
                    skip: skip.map(|attr| attr.span),
                    start: start.map(|attr| attr.span),
                    end: end.map(|attr| attr.span),
                });
                return None;
            }
        };
        Some(kind)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, TyEncodable, TyDecodable, HashStable)]
pub struct CodegenFnAttrFlags(u32);
bitflags::bitflags! {
    impl CodegenFnAttrFlags: u32 {
        /// `#[cold]`: a hint to LLVM that this function, when called, is never on
        /// the hot path.
        const COLD                      = 1 << 0;
        /// `#[rustc_allocator]`: a hint to LLVM that the pointer returned from this
        /// function is never null and the function has no side effects other than allocating.
        const ALLOCATOR                 = 1 << 1;
        /// An indicator that function will never unwind. Will become obsolete
        /// once C-unwind is fully stabilized.
        const NEVER_UNWIND              = 1 << 3;
        /// `#[naked]`: an indicator to LLVM that no function prologue/epilogue
        /// should be generated.
        const NAKED                     = 1 << 4;
        /// `#[no_mangle]`: an indicator that the function's name should be the same
        /// as its symbol.
        const NO_MANGLE                 = 1 << 5;
        /// `#[rustc_std_internal_symbol]`: an indicator that this symbol is a
        /// "weird symbol" for the standard library in that it has slightly
        /// different linkage, visibility, and reachability rules.
        const RUSTC_STD_INTERNAL_SYMBOL = 1 << 6;
        /// `#[thread_local]`: indicates a static is actually a thread local
        /// piece of memory
        const THREAD_LOCAL              = 1 << 8;
        /// `#[used]`: indicates that LLVM can't eliminate this function (but the
        /// linker can!).
        const USED                      = 1 << 9;
        /// `#[track_caller]`: allow access to the caller location
        const TRACK_CALLER              = 1 << 10;
        /// #[ffi_pure]: applies clang's `pure` attribute to a foreign function
        /// declaration.
        const FFI_PURE                  = 1 << 11;
        /// #[ffi_const]: applies clang's `const` attribute to a foreign function
        /// declaration.
        const FFI_CONST                 = 1 << 12;
        // (Bit 13 was used for `#[cmse_nonsecure_entry]`, but is now unused.)
        // (Bit 14 was used for `#[coverage(off)]`, but is now unused.)
        /// `#[used(linker)]`:
        /// indicates that neither LLVM nor the linker will eliminate this function.
        const USED_LINKER               = 1 << 15;
        /// `#[rustc_deallocator]`: a hint to LLVM that the function only deallocates memory.
        const DEALLOCATOR               = 1 << 16;
        /// `#[rustc_reallocator]`: a hint to LLVM that the function only reallocates memory.
        const REALLOCATOR               = 1 << 17;
        /// `#[rustc_allocator_zeroed]`: a hint to LLVM that the function only allocates zeroed memory.
        const ALLOCATOR_ZEROED          = 1 << 18;
        /// `#[no_builtins]`: indicates that disable implicit builtin knowledge of functions for the function.
        const NO_BUILTINS               = 1 << 19;
    }
}
rustc_data_structures::external_bitflags_debug! { CodegenFnAttrFlags }

impl CodegenFnAttrs {
    pub const EMPTY: &'static Self = &Self::new();

    pub const fn new() -> CodegenFnAttrs {
        CodegenFnAttrs {
            flags: CodegenFnAttrFlags::empty(),
            inline: InlineAttr::None,
            optimize: OptimizeAttr::None,
            export_name: None,
            link_name: None,
            link_ordinal: None,
            target_features: vec![],
            linkage: None,
            import_linkage: None,
            link_section: None,
            no_sanitize: SanitizerSet::empty(),
            instruction_set: None,
            alignment: None,
            patchable_function_entry: None,
            skip_short_backtrace: None,
        }
    }

    /// Returns `true` if it looks like this symbol needs to be exported, for example:
    ///
    /// * `#[no_mangle]` is present
    /// * `#[export_name(...)]` is present
    /// * `#[linkage]` is present
    pub fn contains_extern_indicator(&self) -> bool {
        self.flags.contains(CodegenFnAttrFlags::NO_MANGLE)
            || self.export_name.is_some()
            || match self.linkage {
                // These are private, so make sure we don't try to consider
                // them external.
                None | Some(Linkage::Internal | Linkage::Private) => false,
                Some(_) => true,
            }
    }
}
