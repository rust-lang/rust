//! Centralized logic for parsing and validating all attributes used after HIR.
//!
//! History: Check out [#131229](https://github.com/rust-lang/rust/issues/131229).
//! There used to be only one definition of attributes in the compiler: `ast::Attribute`.
//! These were then parsed or validated or both in places distributed all over the compiler.
//!
//! TODO(jdonszelmann): update devguide for best practices on attributes
//! TODO(jdonszelmann): rename to `rustc_attr` in the future, integrating it into this crate.

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(let_chains)]
#![feature(rustdoc_internals)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

use rustc_ast as ast;
use rustc_ast_pretty::pprust;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub enum DiagnosticAttribute {
    // tidy-alphabetical-start
    DoNotRecommend,
    OnUnimplemented,
    // tidy-alphabetical-end
}

// TODO(jdonszelmann): guidelines for when an attribute should be "rustc".
pub enum RustcAttribute {
    // tidy-alphabetical-startt
    AllowConstFnUnstable,
    AllowIncoherentImpl,
    AllowedThroughUnstableModules,
    Coinductive,
    Confusables,
    ConstStable,
    ConstUnstable,
    DenyExplicitImpl,
    HasIncoherentImpls,
    LayoutScalarValidRangeEnd,
    LayoutScalarValidRangeStart,
    LintDiagnostics,
    LintOptDenyFieldAccess,
    LintOptTy,
    LintQueryInstability,
    LintUntrackedQueryInformation,
    MustImplementOneOf,
    ObjectLifetimeDefault,
    PassByValue,
    Promotable,
    PubTransparent,
    SafeIntrinsic,
    // tidy-alphabetical-end
}

/// Attributes represent parsed, *built in* attributes. That means,
/// attributes that are not actually ever expanded. They're instead used as markers,
/// to guide the compilation process in various way in most every stage of the compiler.
/// These are kept around after the AST, into the HIR and further on.
///
/// The word parsed could be a little misleading here, because the parser already parses
/// attributes early on. However, the result, an [`ast::Attribute`](rustc_ast::Attribute)
/// is only parsed at a high level, still containing a token stream in many cases. That is
/// because the structure of the contents varies from attribute to attribute.
/// With a parsed attribute I mean that each attribute is processed individually into a
/// final structure, which on-site (the place where the attribute is useful for, think the
/// the place where `must_use` is checked) little to no extra parsing or validating needs to
/// happen.
pub enum Attribute {
    // tidy-alphabetical-startt
    Allow,
    AllowInternalUnsafe,
    AllowInternalUnstable,
    AutoDiff,
    AutomaticallyDerived,
    Cfg,
    CfgAttr,
    CfiEncoding, // FIXME(cfi_encoding)
    Cold,
    CollapseDebuginfo,
    ConstTrait,
    Coroutine,
    Coverage,
    CustomMir,
    DebuggerVisualizer,
    DefaultLibAllocator,
    Deny,
    Deprecated,
    DeprecatedSafe, // FIXME(deprecated_safe)
    Diagnostic(DiagnosticAttribute),
    Doc,
    Expect,
    ExportName,
    FfiConst,
    FfiPure,
    Forbid,
    Fundamental,
    Ignore,
    Inline,
    InstructionSet, // broken on stable!!!
    Lang,
    Link,
    LinkName,
    LinkOrdinal,
    LinkSection,
    Linkage,
    MacroExport,
    MacroUse,
    Marker,
    MayDangle,
    MustNotSuspend,
    MustUse,
    NeedsAllocator,
    NoImplicitPrelude,
    NoLink,
    NoMangle,
    NoSanitize,
    NonExhaustive,
    OmitGdbPrettyPrinterSection, // FIXME(omit_gdb_pretty_printer_section)
    PanicHandler,
    PatchableFunctionEntry, // FIXME(patchable_function_entry)
    Path,
    Pointee, // FIXME(derive_smart_pointer)
    PreludeImport,
    ProcMacro,
    ProcMacroAttribute,
    ProcMacroDerive,
    Repr,
    Rustc(RustcAttribute),
    Stable,
    Start,
    TargetFeature,
    ThreadLocal,
    TrackCaller,
    Unstable,
    Used,
    Warn,
    WindowsSubsystem, // broken on stable!!!
                      // tidy-alphabetical-end
}

impl Attribute {
    pub fn new(attr: &ast::Attribute) -> Self {
        if let Some(attr) = Self::try_new(attr) {
            attr
        } else {
            unimplemented!("{}", pprust::attribute_to_string(attr));
        }
    }

    pub fn try_new(attr: &ast::Attribute) -> Option<Self> {
        match attr {
            _ => None,
        }
    }
}
