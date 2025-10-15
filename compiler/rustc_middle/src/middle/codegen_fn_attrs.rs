use std::borrow::Cow;

use rustc_abi::Align;
use rustc_hir::attrs::{InlineAttr, InstructionSetAttr, Linkage, OptimizeAttr};
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use rustc_span::Symbol;
use rustc_target::spec::SanitizerSet;

use crate::ty::{InstanceKind, TyCtxt};

impl<'tcx> TyCtxt<'tcx> {
    pub fn codegen_instance_attrs(
        self,
        instance_kind: InstanceKind<'_>,
    ) -> Cow<'tcx, CodegenFnAttrs> {
        let mut attrs = Cow::Borrowed(self.codegen_fn_attrs(instance_kind.def_id()));

        // Drop the `#[naked]` attribute on non-item `InstanceKind`s, like the shims that
        // are generated for indirect function calls.
        if !matches!(instance_kind, InstanceKind::Item(_)) {
            if attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
                attrs.to_mut().flags.remove(CodegenFnAttrFlags::NAKED);
            }
        }

        attrs
    }
}

#[derive(Clone, TyEncodable, TyDecodable, HashStable, Debug)]
pub struct CodegenFnAttrs {
    pub flags: CodegenFnAttrFlags,
    /// Parsed representation of the `#[inline]` attribute
    pub inline: InlineAttr,
    /// Parsed representation of the `#[optimize]` attribute
    pub optimize: OptimizeAttr,
    /// The name this function will be imported/exported under. This can be set
    /// using the `#[export_name = "..."]` or `#[link_name = "..."]` attribute
    /// depending on if this is a function definition or foreign function.
    pub symbol_name: Option<Symbol>,
    /// The `#[link_ordinal = "..."]` attribute, indicating an ordinal an
    /// imported function has in the dynamic library. Note that this must not
    /// be set when `link_name` is set. This is for foreign items with the
    /// "raw-dylib" kind.
    pub link_ordinal: Option<u16>,
    /// The `#[target_feature(enable = "...")]` attribute and the enabled
    /// features (only enabled features are supported right now).
    /// Implied target features have already been applied.
    pub target_features: Vec<TargetFeature>,
    /// Whether the function was declared safe, but has target features
    pub safe_target_features: bool,
    /// The `#[linkage = "..."]` attribute on Rust-defined items and the value we found.
    pub linkage: Option<Linkage>,
    /// The `#[linkage = "..."]` attribute on foreign items and the value we found.
    pub import_linkage: Option<Linkage>,
    /// The `#[link_section = "..."]` attribute, or what executable section this
    /// should be placed in.
    pub link_section: Option<Symbol>,
    /// The `#[sanitize(xyz = "off")]` attribute. Indicates sanitizers for which
    /// instrumentation should be disabled inside the function.
    pub no_sanitize: SanitizerSet,
    /// The `#[instruction_set(set)]` attribute. Indicates if the generated code should
    /// be generated against a specific instruction set. Only usable on architectures which allow
    /// switching between multiple instruction sets.
    pub instruction_set: Option<InstructionSetAttr>,
    /// The `#[align(...)]` attribute. Determines the alignment of the function body.
    // FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
    pub alignment: Option<Align>,
    /// The `#[patchable_function_entry(...)]` attribute. Indicates how many nops should be around
    /// the function entry.
    pub patchable_function_entry: Option<PatchableFunctionEntry>,
    /// The `#[rustc_objc_class = "..."]` attribute.
    pub objc_class: Option<Symbol>,
    /// The `#[rustc_objc_selector = "..."]` attribute.
    pub objc_selector: Option<Symbol>,
}

#[derive(Copy, Clone, Debug, TyEncodable, TyDecodable, HashStable, PartialEq, Eq)]
pub enum TargetFeatureKind {
    /// The feature is implied by another feature, rather than explicitly added by the
    /// `#[target_feature]` attribute
    Implied,
    /// The feature is added by the regular `target_feature` attribute.
    Enabled,
    /// The feature is added by the unsafe `force_target_feature` attribute.
    Forced,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, TyEncodable, TyDecodable, HashStable)]
pub struct TargetFeature {
    /// The name of the target feature (e.g. "avx")
    pub name: Symbol,
    /// The way this feature was enabled.
    pub kind: TargetFeatureKind,
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

#[derive(Clone, Copy, PartialEq, Eq, TyEncodable, TyDecodable, HashStable)]
pub struct CodegenFnAttrFlags(u32);
bitflags::bitflags! {
    impl CodegenFnAttrFlags: u32 {
        /// `#[cold]`: a hint to LLVM that this function, when called, is never on
        /// the hot path.
        const COLD                      = 1 << 0;
        /// `#[rustc_nounwind]`: An indicator that function will never unwind.
        const NEVER_UNWIND              = 1 << 1;
        /// `#[naked]`: an indicator to LLVM that no function prologue/epilogue
        /// should be generated.
        const NAKED                     = 1 << 2;
        /// `#[no_mangle]`: an indicator that the function's name should be the same
        /// as its symbol.
        const NO_MANGLE                 = 1 << 3;
        /// `#[rustc_std_internal_symbol]`: an indicator that this symbol is a
        /// "weird symbol" for the standard library in that it has slightly
        /// different linkage, visibility, and reachability rules.
        const RUSTC_STD_INTERNAL_SYMBOL = 1 << 4;
        /// `#[thread_local]`: indicates a static is actually a thread local
        /// piece of memory
        const THREAD_LOCAL              = 1 << 5;
        /// `#[used(compiler)]`: indicates that LLVM can't eliminate this function (but the
        /// linker can!).
        const USED_COMPILER             = 1 << 6;
        /// `#[used(linker)]`:
        /// indicates that neither LLVM nor the linker will eliminate this function.
        const USED_LINKER               = 1 << 7;
        /// `#[track_caller]`: allow access to the caller location
        const TRACK_CALLER              = 1 << 8;
        /// #[ffi_pure]: applies clang's `pure` attribute to a foreign function
        /// declaration.
        const FFI_PURE                  = 1 << 9;
        /// #[ffi_const]: applies clang's `const` attribute to a foreign function
        /// declaration.
        const FFI_CONST                 = 1 << 10;
        /// `#[rustc_allocator]`: a hint to LLVM that the pointer returned from this
        /// function is never null and the function has no side effects other than allocating.
        const ALLOCATOR                 = 1 << 11;
        /// `#[rustc_deallocator]`: a hint to LLVM that the function only deallocates memory.
        const DEALLOCATOR               = 1 << 12;
        /// `#[rustc_reallocator]`: a hint to LLVM that the function only reallocates memory.
        const REALLOCATOR               = 1 << 13;
        /// `#[rustc_allocator_zeroed]`: a hint to LLVM that the function only allocates zeroed memory.
        const ALLOCATOR_ZEROED          = 1 << 14;
        /// `#[no_builtins]`: indicates that disable implicit builtin knowledge of functions for the function.
        const NO_BUILTINS               = 1 << 15;
        /// Marks foreign items, to make `contains_extern_indicator` cheaper.
        const FOREIGN_ITEM              = 1 << 16;
    }
}
rustc_data_structures::external_bitflags_debug! { CodegenFnAttrFlags }

impl CodegenFnAttrs {
    pub const EMPTY: &'static Self = &Self::new();

    pub const fn new() -> CodegenFnAttrs {
        CodegenFnAttrs {
            flags: CodegenFnAttrFlags::empty(),
            inline: InlineAttr::None,
            optimize: OptimizeAttr::Default,
            symbol_name: None,
            link_ordinal: None,
            target_features: vec![],
            safe_target_features: false,
            linkage: None,
            import_linkage: None,
            link_section: None,
            no_sanitize: SanitizerSet::empty(),
            instruction_set: None,
            alignment: None,
            patchable_function_entry: None,
            objc_class: None,
            objc_selector: None,
        }
    }

    /// Returns `true` if it looks like this symbol needs to be exported, for example:
    ///
    /// * `#[no_mangle]` is present
    /// * `#[export_name(...)]` is present
    /// * `#[linkage]` is present
    ///
    /// Keep this in sync with the logic for the unused_attributes for `#[inline]` lint.
    pub fn contains_extern_indicator(&self) -> bool {
        if self.flags.contains(CodegenFnAttrFlags::FOREIGN_ITEM) {
            return false;
        }

        self.flags.contains(CodegenFnAttrFlags::NO_MANGLE)
            || self.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL)
            || self.symbol_name.is_some()
            || match self.linkage {
                // These are private, so make sure we don't try to consider
                // them external.
                None | Some(Linkage::Internal) => false,
                Some(_) => true,
            }
    }
}
