use crate::mir::mono::Linkage;
use rustc_attr::{InlineAttr, OptimizeAttr};
use rustc_span::symbol::Symbol;

#[derive(Clone, RustcEncodable, RustcDecodable, HashStable)]
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
    pub link_ordinal: Option<usize>,
    /// The `#[target_feature(enable = "...")]` attribute and the enabled
    /// features (only enabled features are supported right now).
    pub target_features: Vec<Symbol>,
    /// The `#[linkage = "..."]` attribute and the value we found.
    pub linkage: Option<Linkage>,
    /// The `#[link_section = "..."]` attribute, or what executable section this
    /// should be placed in.
    pub link_section: Option<Symbol>,
}

bitflags! {
    #[derive(RustcEncodable, RustcDecodable, HashStable)]
    pub struct CodegenFnAttrFlags: u32 {
        /// `#[cold]`: a hint to LLVM that this function, when called, is never on
        /// the hot path.
        const COLD                      = 1 << 0;
        /// `#[rustc_allocator]`: a hint to LLVM that the pointer returned from this
        /// function is never null.
        const ALLOCATOR                 = 1 << 1;
        /// `#[unwind]`: an indicator that this function may unwind despite what
        /// its ABI signature may otherwise imply.
        const UNWIND                    = 1 << 2;
        /// `#[rust_allocator_nounwind]`, an indicator that an imported FFI
        /// function will never unwind. Probably obsolete by recent changes with
        /// #[unwind], but hasn't been removed/migrated yet
        const RUSTC_ALLOCATOR_NOUNWIND  = 1 << 3;
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
        /// `#[no_sanitize(address)]`: disables address sanitizer instrumentation
        const NO_SANITIZE_ADDRESS = 1 << 12;
        /// `#[no_sanitize(memory)]`: disables memory sanitizer instrumentation
        const NO_SANITIZE_MEMORY  = 1 << 13;
        /// `#[no_sanitize(thread)]`: disables thread sanitizer instrumentation
        const NO_SANITIZE_THREAD  = 1 << 14;
        /// All `#[no_sanitize(...)]` attributes.
        const NO_SANITIZE_ANY = Self::NO_SANITIZE_ADDRESS.bits | Self::NO_SANITIZE_MEMORY.bits | Self::NO_SANITIZE_THREAD.bits;
    }
}

impl CodegenFnAttrs {
    pub fn new() -> CodegenFnAttrs {
        CodegenFnAttrs {
            flags: CodegenFnAttrFlags::empty(),
            inline: InlineAttr::None,
            optimize: OptimizeAttr::None,
            export_name: None,
            link_name: None,
            link_ordinal: None,
            target_features: vec![],
            linkage: None,
            link_section: None,
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
                None | Some(Linkage::Internal) | Some(Linkage::Private) => false,
                Some(_) => true,
            }
    }
}
