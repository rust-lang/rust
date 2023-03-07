use crate::mir::mono::Linkage;
use rustc_attr::{InlineAttr, InstructionSetAttr, OptimizeAttr};
use rustc_span::symbol::Symbol;
use rustc_target::spec::SanitizerSet;

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
    pub target_features: Vec<Symbol>,
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
    pub alignment: Option<u32>,
}

bitflags! {
    #[derive(TyEncodable, TyDecodable, HashStable)]
    pub struct CodegenFnAttrFlags: u32 {
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
        /// `#[ffi_returns_twice]`, indicates that an extern function can return
        /// multiple times
        const FFI_RETURNS_TWICE         = 1 << 10;
        /// `#[track_caller]`: allow access to the caller location
        const TRACK_CALLER              = 1 << 11;
        /// #[ffi_pure]: applies clang's `pure` attribute to a foreign function
        /// declaration.
        const FFI_PURE                  = 1 << 12;
        /// #[ffi_const]: applies clang's `const` attribute to a foreign function
        /// declaration.
        const FFI_CONST                 = 1 << 13;
        /// #[cmse_nonsecure_entry]: with a TrustZone-M extension, declare a
        /// function as an entry function from Non-Secure code.
        const CMSE_NONSECURE_ENTRY      = 1 << 14;
        /// `#[no_coverage]`: indicates that the function should be ignored by
        /// the MIR `InstrumentCoverage` pass and not added to the coverage map
        /// during codegen.
        const NO_COVERAGE               = 1 << 15;
        /// `#[used(linker)]`:
        /// indicates that neither LLVM nor the linker will eliminate this function.
        const USED_LINKER               = 1 << 16;
        /// `#[rustc_deallocator]`: a hint to LLVM that the function only deallocates memory.
        const DEALLOCATOR               = 1 << 17;
        /// `#[rustc_reallocator]`: a hint to LLVM that the function only reallocates memory.
        const REALLOCATOR               = 1 << 18;
        /// `#[rustc_allocator_zeroed]`: a hint to LLVM that the function only allocates zeroed memory.
        const ALLOCATOR_ZEROED          = 1 << 19;
    }
}

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
        }
    }

    /// Returns `true` if `#[inline]` or `#[inline(always)]` is present.
    pub fn requests_inline(&self) -> bool {
        match self.inline {
            InlineAttr::Hint | InlineAttr::Always => true,
            InlineAttr::None | InlineAttr::Never => false,
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
