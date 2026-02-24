use std::io::Error;
use std::path::{Path, PathBuf};

use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, DiagSymbolList, Diagnostic, EmissionGuarantee, Level,
    MultiSpan, msg,
};
use rustc_hir::Target;
use rustc_hir::attrs::{MirDialect, MirPhase};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_middle::ty::{MainDefinition, Ty};
use rustc_span::{DUMMY_SP, Ident, Span, Symbol};

use crate::check_attr::ProcMacroKind;
use crate::lang_items::Duplicate;

#[derive(Diagnostic)]
#[diag("`#[diagnostic::do_not_recommend]` can only be placed on trait implementations")]
pub(crate) struct IncorrectDoNotRecommendLocation;

#[derive(Diagnostic)]
#[diag("`#[autodiff]` should be applied to a function")]
pub(crate) struct AutoDiffAttr {
    #[primary_span]
    #[label("not a function")]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[loop_match]` should be applied to a loop")]
pub(crate) struct LoopMatchAttr {
    #[primary_span]
    pub attr_span: Span,
    #[label("not a loop")]
    pub node_span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[const_continue]` should be applied to a break expression")]
pub(crate) struct ConstContinueAttr {
    #[primary_span]
    pub attr_span: Span,
    #[label("not a break expression")]
    pub node_span: Span,
}

#[derive(Diagnostic)]
#[diag("`{$no_mangle_attr}` attribute may not be used in combination with `{$export_name_attr}`")]
pub(crate) struct MixedExportNameAndNoMangle {
    #[label("`{$no_mangle_attr}` is ignored")]
    #[suggestion(
        "remove the `{$no_mangle_attr}` attribute",
        style = "verbose",
        code = "",
        applicability = "machine-applicable"
    )]
    pub no_mangle_span: Span,
    #[note("`{$export_name_attr}` takes precedence")]
    pub export_name_span: Span,
    pub no_mangle_attr: &'static str,
    pub export_name_attr: &'static str,
}

#[derive(Diagnostic)]
#[diag("crate-level attribute should be an inner attribute")]
pub(crate) struct OuterCrateLevelAttr {
    #[subdiagnostic]
    pub suggestion: OuterCrateLevelAttrSuggestion,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("add a `!`", style = "verbose")]
pub(crate) struct OuterCrateLevelAttrSuggestion {
    #[suggestion_part(code = "!")]
    pub bang_position: Span,
}

#[derive(Diagnostic)]
#[diag("crate-level attribute should be in the root module")]
pub(crate) struct InnerCrateLevelAttr;

#[derive(Diagnostic)]
#[diag("`#[non_exhaustive]` can't be used to annotate items with default field values")]
pub(crate) struct NonExhaustiveWithDefaultFieldValues {
    #[primary_span]
    pub attr_span: Span,
    #[label("this struct has default field values")]
    pub defn_span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[doc(alias = \"...\")]` isn't allowed on {$location}")]
pub(crate) struct DocAliasBadLocation<'a> {
    #[primary_span]
    pub span: Span,
    pub location: &'a str,
}

#[derive(Diagnostic)]
#[diag("`#[doc(alias = \"{$attr_str}\"]` is the same as the item's name")]
pub(crate) struct DocAliasNotAnAlias {
    #[primary_span]
    pub span: Span,
    pub attr_str: Symbol,
}

#[derive(Diagnostic)]
#[diag("`#[doc({$attr_name} = \"...\")]` should be used on empty modules")]
pub(crate) struct DocKeywordAttributeEmptyMod {
    #[primary_span]
    pub span: Span,
    pub attr_name: &'static str,
}

#[derive(Diagnostic)]
#[diag("`#[doc({$attr_name} = \"...\")]` should be used on modules")]
pub(crate) struct DocKeywordAttributeNotMod {
    #[primary_span]
    pub span: Span,
    pub attr_name: &'static str,
}

#[derive(Diagnostic)]
#[diag(
    "`#[doc(fake_variadic)]` must be used on the first of a set of tuple or fn pointer trait impls with varying arity"
)]
pub(crate) struct DocFakeVariadicNotValid {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[doc(keyword = \"...\")]` should be used on impl blocks")]
pub(crate) struct DocKeywordOnlyImpl {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[doc(search_unbox)]` should be used on generic structs and enums")]
pub(crate) struct DocSearchUnboxInvalid {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("conflicting doc inlining attributes")]
#[help("remove one of the conflicting attributes")]
pub(crate) struct DocInlineConflict {
    #[primary_span]
    pub spans: MultiSpan,
}

#[derive(Diagnostic)]
#[diag("this attribute can only be applied to a `use` item")]
#[note(
    "read <https://doc.rust-lang.org/nightly/rustdoc/the-doc-attribute.html#inline-and-no_inline> for more information"
)]
pub(crate) struct DocInlineOnlyUse {
    #[label("only applicable on `use` items")]
    pub attr_span: Span,
    #[label("not a `use` item")]
    pub item_span: Span,
}

#[derive(Diagnostic)]
#[diag("this attribute can only be applied to an `extern crate` item")]
#[note(
    "read <https://doc.rust-lang.org/unstable-book/language-features/doc-masked.html> for more information"
)]
pub(crate) struct DocMaskedOnlyExternCrate {
    #[label("only applicable on `extern crate` items")]
    pub attr_span: Span,
    #[label("not an `extern crate` item")]
    pub item_span: Span,
}

#[derive(Diagnostic)]
#[diag("this attribute cannot be applied to an `extern crate self` item")]
pub(crate) struct DocMaskedNotExternCrateSelf {
    #[label("not applicable on `extern crate self` items")]
    pub attr_span: Span,
    #[label("`extern crate self` defined here")]
    pub item_span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[ffi_const]` function cannot be `#[ffi_pure]`", code = E0757)]
pub(crate) struct BothFfiConstAndPure {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag("attribute should be applied to an `extern` block with non-Rust ABI")]
#[warning(
    "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
)]
pub(crate) struct Link {
    #[label("not an `extern` block")]
    pub span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("#[rustc_legacy_const_generics] functions must only have const generics")]
pub(crate) struct RustcLegacyConstGenericsOnly {
    #[primary_span]
    pub attr_span: Span,
    #[label("non-const generic parameter")]
    pub param_span: Span,
}

#[derive(Diagnostic)]
#[diag("#[rustc_legacy_const_generics] must have one index for each generic parameter")]
pub(crate) struct RustcLegacyConstGenericsIndex {
    #[primary_span]
    pub attr_span: Span,
    #[label("generic parameters")]
    pub generics_span: Span,
}

#[derive(Diagnostic)]
#[diag("index exceeds number of arguments")]
pub(crate) struct RustcLegacyConstGenericsIndexExceed {
    #[primary_span]
    #[label(
        "there {$arg_count ->
            [one] is
            *[other] are
        } only {$arg_count} {$arg_count ->
            [one] argument
            *[other] arguments
        }"
    )]
    pub span: Span,
    pub arg_count: usize,
}

#[derive(Diagnostic)]
#[diag("conflicting representation hints", code = E0566)]
pub(crate) struct ReprConflicting {
    #[primary_span]
    pub hint_spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("alignment must not be greater than `isize::MAX` bytes", code = E0589)]
#[note("`isize::MAX` is {$size} for the current target")]
pub(crate) struct InvalidReprAlignForTarget {
    #[primary_span]
    pub span: Span,
    pub size: u64,
}

#[derive(Diagnostic)]
#[diag("conflicting representation hints", code = E0566)]
pub(crate) struct ReprConflictingLint;

#[derive(Diagnostic)]
#[diag("attribute should be applied to a macro")]
pub(crate) struct MacroOnlyAttribute {
    #[primary_span]
    pub attr_span: Span,
    #[label("not a macro")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("couldn't read {$file}: {$error}")]
pub(crate) struct DebugVisualizerUnreadable<'a> {
    #[primary_span]
    pub span: Span,
    pub file: &'a Path,
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag("attribute should be applied to `const fn`")]
pub(crate) struct RustcAllowConstFnUnstable {
    #[primary_span]
    pub attr_span: Span,
    #[label("not a `const fn`")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("attribute should be applied to `#[repr(transparent)]` types")]
pub(crate) struct RustcPubTransparent {
    #[primary_span]
    pub attr_span: Span,
    #[label("not a `#[repr(transparent)]` type")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("attribute cannot be applied to a `async`, `gen` or `async gen` function")]
pub(crate) struct RustcForceInlineCoro {
    #[primary_span]
    pub attr_span: Span,
    #[label("`async`, `gen` or `async gen` function")]
    pub span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum MacroExport {
    #[diag("`#[macro_export]` has no effect on declarative macro definitions")]
    #[note("declarative macros follow the same exporting rules as regular items")]
    OnDeclMacro,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnusedNote {
    #[note("attribute `{$name}` with an empty list has no effect")]
    EmptyList { name: Symbol },
    #[note("attribute `{$name}` without any lints has no effect")]
    NoLints { name: Symbol },
    #[note("`default_method_body_is_const` has been replaced with `const` on traits")]
    DefaultMethodBodyConst,
    #[note(
        "the `linker_messages` lint can only be controlled at the root of a crate that needs to be linked"
    )]
    LinkerMessagesBinaryCrateOnly,
}

#[derive(Diagnostic)]
#[diag("unused attribute")]
pub(crate) struct Unused {
    #[suggestion("remove this attribute", code = "", applicability = "machine-applicable")]
    pub attr_span: Span,
    #[subdiagnostic]
    pub note: UnusedNote,
}

#[derive(Diagnostic)]
#[diag("attribute should be applied to function or closure", code = E0518)]
pub(crate) struct NonExportedMacroInvalidAttrs {
    #[primary_span]
    #[label("not a function or closure")]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[may_dangle]` must be applied to a lifetime or type generic parameter in `Drop` impl")]
pub(crate) struct InvalidMayDangle {
    #[primary_span]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag("unused attribute")]
pub(crate) struct UnusedDuplicate {
    #[suggestion("remove this attribute", code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note("attribute also specified here")]
    pub other: Span,
    #[warning(
        "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
    )]
    pub warning: bool,
}

#[derive(Diagnostic)]
#[diag("multiple `{$name}` attributes")]
pub(crate) struct UnusedMultiple {
    #[primary_span]
    #[suggestion("remove this attribute", code = "", applicability = "machine-applicable")]
    pub this: Span,
    #[note("attribute also specified here")]
    pub other: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("this `#[deprecated]` annotation has no effect")]
pub(crate) struct DeprecatedAnnotationHasNoEffect {
    #[suggestion(
        "remove the unnecessary deprecation attribute",
        applicability = "machine-applicable",
        code = ""
    )]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unknown external lang item: `{$lang_item}`", code = E0264)]
pub(crate) struct UnknownExternLangItem {
    #[primary_span]
    pub span: Span,
    pub lang_item: Symbol,
}

#[derive(Diagnostic)]
#[diag("`#[panic_handler]` function required, but not found")]
pub(crate) struct MissingPanicHandler;

#[derive(Diagnostic)]
#[diag("unwinding panics are not supported without std")]
#[help("using nightly cargo, use -Zbuild-std with panic=\"abort\" to avoid unwinding")]
#[note(
    "since the core library is usually precompiled with panic=\"unwind\", rebuilding your crate with panic=\"abort\" may not be enough to fix the problem"
)]
pub(crate) struct PanicUnwindWithoutStd;

#[derive(Diagnostic)]
#[diag("lang item required, but not found: `{$name}`")]
#[note(
    "this can occur when a binary crate with `#![no_std]` is compiled for a target where `{$name}` is defined in the standard library"
)]
#[help(
    "you may be able to compile for a target that doesn't need `{$name}`, specify a target with `--target` or in `.cargo/config`"
)]
pub(crate) struct MissingLangItem {
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(
    "{$name ->
    [panic_impl] `#[panic_handler]`
    *[other] `{$name}` lang item
} function is not allowed to have `#[track_caller]`"
)]
pub(crate) struct LangItemWithTrackCaller {
    #[primary_span]
    pub attr_span: Span,
    pub name: Symbol,
    #[label(
        "{$name ->
            [panic_impl] `#[panic_handler]`
            *[other] `{$name}` lang item
        } function is not allowed to have `#[track_caller]`"
    )]
    pub sig_span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "{$name ->
        [panic_impl] `#[panic_handler]`
        *[other] `{$name}` lang item
    } function is not allowed to have `#[target_feature]`"
)]
pub(crate) struct LangItemWithTargetFeature {
    #[primary_span]
    pub attr_span: Span,
    pub name: Symbol,
    #[label(
        "{$name ->
            [panic_impl] `#[panic_handler]`
            *[other] `{$name}` lang item
        } function is not allowed to have `#[target_feature]`"
    )]
    pub sig_span: Span,
}

#[derive(Diagnostic)]
#[diag("`{$name}` lang item must be applied to a {$expected_target}", code = E0718)]
pub(crate) struct LangItemOnIncorrectTarget {
    #[primary_span]
    #[label("attribute should be applied to a {$expected_target}, not a {$actual_target}")]
    pub span: Span,
    pub name: Symbol,
    pub expected_target: Target,
    pub actual_target: Target,
}

pub(crate) struct InvalidAttrAtCrateLevel {
    pub span: Span,
    pub sugg_span: Option<Span>,
    pub name: Symbol,
    pub item: Option<ItemFollowingInnerAttr>,
}

#[derive(Clone, Copy)]
pub(crate) struct ItemFollowingInnerAttr {
    pub span: Span,
    pub kind: &'static str,
}

impl<G: EmissionGuarantee> Diagnostic<'_, G> for InvalidAttrAtCrateLevel {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let mut diag =
            Diag::new(dcx, level, msg!("`{$name}` attribute cannot be used at crate level"));
        diag.span(self.span);
        diag.arg("name", self.name);
        // Only emit an error with a suggestion if we can create a string out
        // of the attribute span
        if let Some(span) = self.sugg_span {
            diag.span_suggestion_verbose(
                span,
                msg!("perhaps you meant to use an outer attribute"),
                String::new(),
                Applicability::MachineApplicable,
            );
        }
        if let Some(item) = self.item {
            diag.arg("kind", item.kind);
            diag.span_label(item.span, msg!("the inner attribute doesn't annotate this {$kind}"));
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag("duplicate diagnostic item in crate `{$crate_name}`: `{$name}`")]
pub(crate) struct DuplicateDiagnosticItemInCrate {
    #[primary_span]
    pub duplicate_span: Option<Span>,
    #[note("the diagnostic item is first defined here")]
    pub orig_span: Option<Span>,
    #[note("the diagnostic item is first defined in crate `{$orig_crate_name}`")]
    pub different_crates: bool,
    pub crate_name: Symbol,
    pub orig_crate_name: Symbol,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag("abi: {$abi}")]
pub(crate) struct LayoutAbi {
    #[primary_span]
    pub span: Span,
    pub abi: String,
}

#[derive(Diagnostic)]
#[diag("align: {$align}")]
pub(crate) struct LayoutAlign {
    #[primary_span]
    pub span: Span,
    pub align: String,
}

#[derive(Diagnostic)]
#[diag("size: {$size}")]
pub(crate) struct LayoutSize {
    #[primary_span]
    pub span: Span,
    pub size: String,
}

#[derive(Diagnostic)]
#[diag("homogeneous_aggregate: {$homogeneous_aggregate}")]
pub(crate) struct LayoutHomogeneousAggregate {
    #[primary_span]
    pub span: Span,
    pub homogeneous_aggregate: String,
}

#[derive(Diagnostic)]
#[diag("layout_of({$normalized_ty}) = {$ty_layout}")]
pub(crate) struct LayoutOf<'tcx> {
    #[primary_span]
    pub span: Span,
    pub normalized_ty: Ty<'tcx>,
    pub ty_layout: String,
}

#[derive(Diagnostic)]
#[diag("fn_abi_of({$fn_name}) = {$fn_abi}")]
pub(crate) struct AbiOf {
    #[primary_span]
    pub span: Span,
    pub fn_name: Symbol,
    pub fn_abi: String,
}

#[derive(Diagnostic)]
#[diag(
    "ABIs are not compatible
    left ABI = {$left}
    right ABI = {$right}"
)]
pub(crate) struct AbiNe {
    #[primary_span]
    pub span: Span,
    pub left: String,
    pub right: String,
}

#[derive(Diagnostic)]
#[diag(
    "`#[rustc_abi]` can only be applied to function items, type aliases, and associated functions"
)]
pub(crate) struct AbiInvalidAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unrecognized argument")]
pub(crate) struct UnrecognizedArgument {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("feature `{$feature}` is declared stable since {$since}, but was previously declared stable since {$prev_since}", code = E0711)]
pub(crate) struct FeatureStableTwice {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub since: Symbol,
    pub prev_since: Symbol,
}

#[derive(Diagnostic)]
#[diag("feature `{$feature}` is declared {$declared}, but was previously declared {$prev_declared}", code = E0711)]
pub(crate) struct FeaturePreviouslyDeclared<'a> {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub declared: &'a str,
    pub prev_declared: &'a str,
}

#[derive(Diagnostic)]
#[diag("multiple functions with a `#[rustc_main]` attribute", code = E0137)]
pub(crate) struct MultipleRustcMain {
    #[primary_span]
    pub span: Span,
    #[label("first `#[rustc_main]` function")]
    pub first: Span,
    #[label("additional `#[rustc_main]` function")]
    pub additional: Span,
}

#[derive(Diagnostic)]
#[diag("the `main` function cannot be declared in an `extern` block")]
pub(crate) struct ExternMain {
    #[primary_span]
    pub span: Span,
}

pub(crate) struct NoMainErr {
    pub sp: Span,
    pub crate_name: Symbol,
    pub has_filename: bool,
    pub filename: PathBuf,
    pub file_empty: bool,
    pub non_main_fns: Vec<Span>,
    pub main_def_opt: Option<MainDefinition>,
    pub add_teach_note: bool,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for NoMainErr {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let mut diag =
            Diag::new(dcx, level, msg!("`main` function not found in crate `{$crate_name}`"));
        diag.span(DUMMY_SP);
        diag.code(E0601);
        diag.arg("crate_name", self.crate_name);
        diag.arg("filename", self.filename);
        diag.arg("has_filename", self.has_filename);
        let note = if !self.non_main_fns.is_empty() {
            for &span in &self.non_main_fns {
                diag.span_note(span, msg!("here is a function named `main`"));
            }
            diag.note(msg!(
                "you have one or more functions named `main` not defined at the crate level"
            ));
            diag.help(msg!("consider moving the `main` function definitions"));
            // There were some functions named `main` though. Try to give the user a hint.
            msg!(
                "the main function must be defined at the crate level{$has_filename ->
                    [true] {\" \"}(in `{$filename}`)
                    *[false] {\"\"}
                }"
            )
        } else if self.has_filename {
            msg!("consider adding a `main` function to `{$filename}`")
        } else {
            msg!("consider adding a `main` function at the crate level")
        };
        if self.file_empty {
            diag.note(note);
        } else {
            diag.span(self.sp.shrink_to_hi());
            diag.span_label(self.sp.shrink_to_hi(), note);
        }

        if let Some(main_def) = self.main_def_opt
            && main_def.opt_fn_def_id().is_none()
        {
            // There is something at `crate::main`, but it is not a function definition.
            diag.span_label(main_def.span, msg!("non-function item at `crate::main` is found"));
        }

        if self.add_teach_note {
            diag.note(msg!("if you don't know the basics of Rust, you can go look to the Rust Book to get started: https://doc.rust-lang.org/book/"));
        }
        diag
    }
}

pub(crate) struct DuplicateLangItem {
    pub local_span: Option<Span>,
    pub lang_item_name: Symbol,
    pub crate_name: Symbol,
    pub dependency_of: Option<Symbol>,
    pub is_local: bool,
    pub path: String,
    pub first_defined_span: Option<Span>,
    pub orig_crate_name: Option<Symbol>,
    pub orig_dependency_of: Option<Symbol>,
    pub orig_is_local: bool,
    pub orig_path: String,
    pub(crate) duplicate: Duplicate,
}

impl<G: EmissionGuarantee> Diagnostic<'_, G> for DuplicateLangItem {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let mut diag = Diag::new(
            dcx,
            level,
            match self.duplicate {
                Duplicate::Plain => msg!("found duplicate lang item `{$lang_item_name}`"),
                Duplicate::Crate => {
                    msg!("duplicate lang item in crate `{$crate_name}`: `{$lang_item_name}`")
                }
                Duplicate::CrateDepends => msg!(
                    "duplicate lang item in crate `{$crate_name}` (which `{$dependency_of}` depends on): `{$lang_item_name}`"
                ),
            },
        );
        diag.code(E0152);
        diag.arg("lang_item_name", self.lang_item_name);
        diag.arg("crate_name", self.crate_name);
        if let Some(dependency_of) = self.dependency_of {
            diag.arg("dependency_of", dependency_of);
        }
        diag.arg("path", self.path);
        if let Some(orig_crate_name) = self.orig_crate_name {
            diag.arg("orig_crate_name", orig_crate_name);
        }
        if let Some(orig_dependency_of) = self.orig_dependency_of {
            diag.arg("orig_dependency_of", orig_dependency_of);
        }
        diag.arg("orig_path", self.orig_path);
        if let Some(span) = self.local_span {
            diag.span(span);
        }
        if let Some(span) = self.first_defined_span {
            diag.span_note(span, msg!("the lang item is first defined here"));
        } else {
            if self.orig_dependency_of.is_none() {
                diag.note(msg!("the lang item is first defined in crate `{$orig_crate_name}`"));
            } else {
                diag.note(msg!("the lang item is first defined in crate `{$orig_crate_name}` (which `{$orig_dependency_of}` depends on)"));
            }

            if self.orig_is_local {
                diag.note(msg!("first definition in the local crate (`{$orig_crate_name}`)"));
            } else {
                diag.note(msg!(
                    "first definition in `{$orig_crate_name}` loaded from {$orig_path}"
                ));
            }

            if self.is_local {
                diag.note(msg!("second definition in the local crate (`{$crate_name}`)"));
            } else {
                diag.note(msg!("second definition in `{$crate_name}` loaded from {$path}"));
            }
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag("`{$name}` lang item must be applied to a {$kind} with {$at_least ->
        [true] at least {$num}
        *[false] {$num}
    } generic {$num ->
        [one] argument
        *[other] arguments
    }", code = E0718)]
pub(crate) struct IncorrectTarget<'a> {
    #[primary_span]
    pub span: Span,
    #[label(
        "this {$kind} has {$actual_num} generic {$actual_num ->
            [one] argument
            *[other] arguments
        }"
    )]
    pub generics_span: Span,
    pub name: &'a str, // cannot be symbol because it renders e.g. `r#fn` instead of `fn`
    pub kind: &'static str,
    pub num: usize,
    pub actual_num: usize,
    pub at_least: bool,
}

#[derive(Diagnostic)]
#[diag("lang items are not allowed in stable dylibs")]
pub(crate) struct IncorrectCrateType {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "useless assignment of {$is_field_assign ->
        [true] field
        *[false] variable
    } of type `{$ty}` to itself"
)]
pub(crate) struct UselessAssignment<'a> {
    pub is_field_assign: bool,
    pub ty: Ty<'a>,
}

#[derive(Diagnostic)]
#[diag("`#[inline]` is ignored on externally exported functions")]
#[help(
    "externally exported functions are functions with `#[no_mangle]`, `#[export_name]`, or `#[linkage]`"
)]
pub(crate) struct InlineIgnoredForExported;

#[derive(Diagnostic)]
#[diag("{$repr}")]
pub(crate) struct ObjectLifetimeErr {
    #[primary_span]
    pub span: Span,
    pub repr: String,
}

#[derive(Diagnostic)]
pub(crate) enum AttrApplication {
    #[diag("attribute should be applied to an enum", code = E0517)]
    Enum {
        #[primary_span]
        hint_span: Span,
        #[label("not an enum")]
        span: Span,
    },
    #[diag("attribute should be applied to a struct", code = E0517)]
    Struct {
        #[primary_span]
        hint_span: Span,
        #[label("not a struct")]
        span: Span,
    },
    #[diag("attribute should be applied to a struct or union", code = E0517)]
    StructUnion {
        #[primary_span]
        hint_span: Span,
        #[label("not a struct or union")]
        span: Span,
    },
    #[diag("attribute should be applied to a struct, enum, or union", code = E0517)]
    StructEnumUnion {
        #[primary_span]
        hint_span: Span,
        #[label("not a struct, enum, or union")]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("transparent {$target} cannot have other repr hints", code = E0692)]
pub(crate) struct TransparentIncompatible {
    #[primary_span]
    pub hint_spans: Vec<Span>,
    pub target: String,
}

#[derive(Diagnostic)]
#[diag("deprecated attribute must be paired with either stable or unstable attribute", code = E0549)]
pub(crate) struct DeprecatedAttribute {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("this stability annotation is useless")]
pub(crate) struct UselessStability {
    #[primary_span]
    #[label("useless stability annotation")]
    pub span: Span,
    #[label("the stability attribute annotates this item")]
    pub item_sp: Span,
}

#[derive(Diagnostic)]
#[diag("an API can't be stabilized after it is deprecated")]
pub(crate) struct CannotStabilizeDeprecated {
    #[primary_span]
    #[label("invalid version")]
    pub span: Span,
    #[label("the stability attribute annotates this item")]
    pub item_sp: Span,
}

#[derive(Diagnostic)]
#[diag("can't mark as unstable using an already stable feature")]
pub(crate) struct UnstableAttrForAlreadyStableFeature {
    #[primary_span]
    #[label("this feature is already stable")]
    #[help("consider removing the attribute")]
    pub attr_span: Span,
    #[label("the stability attribute annotates this item")]
    pub item_span: Span,
}

#[derive(Diagnostic)]
#[diag("{$descr} has missing stability attribute")]
pub(crate) struct MissingStabilityAttr<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: &'a str,
}

#[derive(Diagnostic)]
#[diag("{$descr} has missing const stability attribute")]
pub(crate) struct MissingConstStabAttr<'a> {
    #[primary_span]
    pub span: Span,
    pub descr: &'a str,
}

#[derive(Diagnostic)]
#[diag("trait implementations cannot be const stable yet")]
#[note("see issue #143874 <https://github.com/rust-lang/rust/issues/143874> for more information")]
pub(crate) struct TraitImplConstStable {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("const stability on the impl does not match the const stability on the trait")]
pub(crate) struct TraitImplConstStabilityMismatch {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub impl_stability: ImplConstStability,
    #[subdiagnostic]
    pub trait_stability: TraitConstStability,
}

#[derive(Subdiagnostic)]
pub(crate) enum TraitConstStability {
    #[note("...but the trait is stable")]
    Stable {
        #[primary_span]
        span: Span,
    },
    #[note("...but the trait is unstable")]
    Unstable {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum ImplConstStability {
    #[note("this impl is (implicitly) stable...")]
    Stable {
        #[primary_span]
        span: Span,
    },
    #[note("this impl is unstable...")]
    Unstable {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("unknown feature `{$feature}`", code = E0635)]
pub(crate) struct UnknownFeature {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    #[subdiagnostic]
    pub suggestion: Option<MisspelledFeature>,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "there is a feature with a similar name: `{$actual_name}`",
    style = "verbose",
    code = "{actual_name}",
    applicability = "maybe-incorrect"
)]
pub(crate) struct MisspelledFeature {
    #[primary_span]
    pub span: Span,
    pub actual_name: Symbol,
}

#[derive(Diagnostic)]
#[diag("feature `{$alias}` has been renamed to `{$feature}`", code = E0635)]
pub(crate) struct RenamedFeature {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub alias: Symbol,
}

#[derive(Diagnostic)]
#[diag("feature `{$implied_by}` implying `{$feature}` does not exist")]
pub(crate) struct ImpliedFeatureNotExist {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
    pub implied_by: Symbol,
}

#[derive(Diagnostic)]
#[diag("the feature `{$feature}` has already been enabled", code = E0636)]
pub(crate) struct DuplicateFeatureErr {
    #[primary_span]
    pub span: Span,
    pub feature: Symbol,
}

#[derive(Diagnostic)]
#[diag(
    "attributes `#[rustc_const_unstable]`, `#[rustc_const_stable]` and `#[rustc_const_stable_indirect]` require the function or method to be `const`"
)]
pub(crate) struct MissingConstErr {
    #[primary_span]
    #[help("make the function or method const")]
    pub fn_sig_span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "attribute `#[rustc_const_stable]` can only be applied to functions that are declared `#[stable]`"
)]
pub(crate) struct ConstStableNotStable {
    #[primary_span]
    pub fn_sig_span: Span,
    #[label("attribute specified here")]
    pub const_span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum MultipleDeadCodes<'tcx> {
    #[diag(
        "{ $multiple ->
            *[true] multiple {$descr}s are
            [false] { $num ->
                [one] {$descr} {$name_list} is
                *[other] {$descr}s {$name_list} are
            }
        } never {$participle}"
    )]
    DeadCodes {
        multiple: bool,
        num: usize,
        descr: &'tcx str,
        participle: &'tcx str,
        name_list: DiagSymbolList,
        #[subdiagnostic]
        // only on DeadCodes since it's never a problem for tuple struct fields
        enum_variants_with_same_name: Vec<EnumVariantSameName<'tcx>>,
        #[subdiagnostic]
        parent_info: Option<ParentInfo<'tcx>>,
        #[subdiagnostic]
        ignored_derived_impls: Option<IgnoredDerivedImpls>,
    },
    #[diag(
        "{ $multiple ->
            *[true] multiple {$descr}s are
            [false] { $num ->
                [one] {$descr} {$name_list} is
                *[other] {$descr}s {$name_list} are
            }
        } never {$participle}"
    )]
    UnusedTupleStructFields {
        multiple: bool,
        num: usize,
        descr: &'tcx str,
        participle: &'tcx str,
        name_list: DiagSymbolList,
        #[subdiagnostic]
        change_fields_suggestion: ChangeFields,
        #[subdiagnostic]
        parent_info: Option<ParentInfo<'tcx>>,
        #[subdiagnostic]
        ignored_derived_impls: Option<IgnoredDerivedImpls>,
    },
}

#[derive(Subdiagnostic)]
#[note(
    "it is impossible to refer to the {$dead_descr} `{$dead_name}` because it is shadowed by this enum variant with the same name"
)]
pub(crate) struct EnumVariantSameName<'tcx> {
    #[primary_span]
    pub variant_span: Span,
    pub dead_name: Symbol,
    pub dead_descr: &'tcx str,
}

#[derive(Subdiagnostic)]
#[label(
    "{$num ->
        [one] {$descr}
        *[other] {$descr}s
    } in this {$parent_descr}"
)]
pub(crate) struct ParentInfo<'tcx> {
    pub num: usize,
    pub descr: &'tcx str,
    pub parent_descr: &'tcx str,
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[note(
    "`{$name}` has {$trait_list_len ->
        [one] a derived impl
        *[other] derived impls
    } for the {$trait_list_len ->
        [one] trait {$trait_list}, but this is
        *[other] traits {$trait_list}, but these are
    } intentionally ignored during dead code analysis"
)]
pub(crate) struct IgnoredDerivedImpls {
    pub name: Symbol,
    pub trait_list: DiagSymbolList,
    pub trait_list_len: usize,
}

#[derive(Subdiagnostic)]
pub(crate) enum ChangeFields {
    #[multipart_suggestion(
        "consider changing the { $num ->
            [one] field
            *[other] fields
        } to be of unit type to suppress this warning while preserving the field numbering, or remove the { $num ->
            [one] field
            *[other] fields
        }",
        applicability = "has-placeholders"
    )]
    ChangeToUnitTypeOrRemove {
        num: usize,
        #[suggestion_part(code = "()")]
        spans: Vec<Span>,
    },
    #[help(
        "consider removing { $num ->
            [one] this
            *[other] these
        } { $num ->
            [one] field
            *[other] fields
        }"
    )]
    Remove { num: usize },
}

#[derive(Diagnostic)]
#[diag("{$kind} has incorrect signature")]
pub(crate) struct ProcMacroBadSig {
    #[primary_span]
    pub span: Span,
    pub kind: ProcMacroKind,
}

#[derive(Diagnostic)]
#[diag(
    "the feature `{$feature}` has been stable since {$since} and no longer requires an attribute to enable"
)]
pub(crate) struct UnnecessaryStableFeature {
    pub feature: Symbol,
    pub since: Symbol,
}

#[derive(Diagnostic)]
#[diag(
    "the feature `{$feature}` has been partially stabilized since {$since} and is succeeded by the feature `{$implies}`"
)]
pub(crate) struct UnnecessaryPartialStableFeature {
    #[suggestion(
        "if you are using features which are still unstable, change to using `{$implies}`",
        code = "{implies}",
        applicability = "maybe-incorrect"
    )]
    pub span: Span,
    #[suggestion(
        "if you are using features which are now stable, remove this line",
        code = "",
        applicability = "maybe-incorrect"
    )]
    pub line: Span,
    pub feature: Symbol,
    pub since: Symbol,
    pub implies: Symbol,
}

#[derive(Diagnostic)]
#[diag("an `#[unstable]` annotation here has no effect")]
#[note("see issue #55436 <https://github.com/rust-lang/rust/issues/55436> for more information")]
pub(crate) struct IneffectiveUnstableImpl;

#[derive(Diagnostic)]
#[diag("sanitize attribute not allowed here")]
pub(crate) struct SanitizeAttributeNotAllowed {
    #[primary_span]
    pub attr_span: Span,
    #[label("not a function, impl block, or module")]
    pub not_fn_impl_mod: Option<Span>,
    #[label("function has no body")]
    pub no_body: Option<Span>,
    #[help("sanitize attribute can be applied to a function (with body), impl block, or module")]
    pub help: (),
}

// FIXME(jdonszelmann): move back to rustc_attr
#[derive(Diagnostic)]
#[diag(
    "`const_stable_indirect` attribute does not make sense on `rustc_const_stable` function, its behavior is already implied"
)]
pub(crate) struct RustcConstStableIndirectPairing {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("most attributes are not supported in `where` clauses")]
#[help("only `#[cfg]` and `#[cfg_attr]` are supported")]
pub(crate) struct UnsupportedAttributesInWhere {
    #[primary_span]
    pub span: MultiSpan,
}

#[derive(Diagnostic)]
pub(crate) enum UnexportableItem<'a> {
    #[diag("{$descr}'s are not exportable")]
    Item {
        #[primary_span]
        span: Span,
        descr: &'a str,
    },

    #[diag("generic functions are not exportable")]
    GenericFn(#[primary_span] Span),

    #[diag("only functions with \"C\" ABI are exportable")]
    FnAbi(#[primary_span] Span),

    #[diag("types with unstable layout are not exportable")]
    TypeRepr(#[primary_span] Span),

    #[diag("{$desc} with `#[export_stable]` attribute uses type `{$ty}`, which is not exportable")]
    TypeInInterface {
        #[primary_span]
        span: Span,
        desc: &'a str,
        ty: &'a str,
        #[label("not exportable")]
        ty_span: Span,
    },

    #[diag("private items are not exportable")]
    PrivItem {
        #[primary_span]
        span: Span,
        #[note("is only usable at visibility `{$vis_descr}`")]
        vis_note: Span,
        vis_descr: &'a str,
    },

    #[diag("ADT types with private fields are not exportable")]
    AdtWithPrivFields {
        #[primary_span]
        span: Span,
        #[note("`{$field_name}` is private")]
        vis_note: Span,
        field_name: &'a str,
    },
}

#[derive(Diagnostic)]
#[diag("`#[repr(align(...))]` is not supported on {$item}")]
pub(crate) struct ReprAlignShouldBeAlign {
    #[primary_span]
    #[help("use `#[rustc_align(...)]` instead")]
    pub span: Span,
    pub item: &'static str,
}

#[derive(Diagnostic)]
#[diag("`#[repr(align(...))]` is not supported on {$item}")]
pub(crate) struct ReprAlignShouldBeAlignStatic {
    #[primary_span]
    #[help("use `#[rustc_align_static(...)]` instead")]
    pub span: Span,
    pub item: &'static str,
}

#[derive(Diagnostic)]
#[diag("`dialect` key required")]
pub(crate) struct CustomMirPhaseRequiresDialect {
    #[primary_span]
    pub attr_span: Span,
    #[label("`phase` argument requires a `dialect` argument")]
    pub phase_span: Span,
}

#[derive(Diagnostic)]
#[diag("the {$dialect} dialect is not compatible with the {$phase} phase")]
pub(crate) struct CustomMirIncompatibleDialectAndPhase {
    pub dialect: MirDialect,
    pub phase: MirPhase,
    #[primary_span]
    pub attr_span: Span,
    #[label("this dialect...")]
    pub dialect_span: Span,
    #[label("... is not compatible with this phase")]
    pub phase_span: Span,
}

#[derive(Diagnostic)]
#[diag("`eii_macro_for` is only valid on functions")]
pub(crate) struct EiiImplNotFunction {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[{$name}]` is unsafe to implement")]
pub(crate) struct EiiImplRequiresUnsafe {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    #[subdiagnostic]
    pub suggestion: EiiImplRequiresUnsafeSuggestion,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("wrap the attribute in `unsafe(...)`", applicability = "machine-applicable")]
pub(crate) struct EiiImplRequiresUnsafeSuggestion {
    #[suggestion_part(code = "unsafe(")]
    pub left: Span,
    #[suggestion_part(code = ")")]
    pub right: Span,
}

#[derive(Diagnostic)]
#[diag("`#[{$name}]` is not allowed to have `#[track_caller]`")]
pub(crate) struct EiiWithTrackCaller {
    #[primary_span]
    pub attr_span: Span,
    pub name: Symbol,
    #[label("`#[{$name}]` is not allowed to have `#[track_caller]`")]
    pub sig_span: Span,
}

#[derive(Diagnostic)]
#[diag("`#[{$name}]` required, but not found")]
pub(crate) struct EiiWithoutImpl {
    #[primary_span]
    #[label("expected because `#[{$name}]` was declared here in crate `{$decl_crate_name}`")]
    pub span: Span,
    pub name: Symbol,

    pub current_crate_name: Symbol,
    pub decl_crate_name: Symbol,
    #[help(
        "expected at least one implementation in crate `{$current_crate_name}` or any of its dependencies"
    )]
    pub help: (),
}

#[derive(Diagnostic)]
#[diag("multiple implementations of `#[{$name}]`")]
pub(crate) struct DuplicateEiiImpls {
    pub name: Symbol,

    #[primary_span]
    #[label("first implemented here in crate `{$first_crate}`")]
    pub first_span: Span,
    pub first_crate: Symbol,

    #[label("also implemented here in crate `{$second_crate}`")]
    pub second_span: Span,
    pub second_crate: Symbol,

    #[note("in addition to these two, { $num_additional_crates ->
        [one] another implementation was found in crate {$additional_crate_names}
        *[other] more implementations were also found in the following crates: {$additional_crate_names}
    }")]
    pub additional_crates: Option<()>,

    pub num_additional_crates: usize,
    pub additional_crate_names: String,

    #[help(
        "an \"externally implementable item\" can only have a single implementation in the final artifact. When multiple implementations are found, also in different crates, they conflict"
    )]
    pub help: (),
}

#[derive(Diagnostic)]
#[diag("function doesn't have a default implementation")]
pub(crate) struct FunctionNotHaveDefaultImplementation {
    #[primary_span]
    pub span: Span,
    #[note("required by this annotation")]
    pub note_span: Span,
}

#[derive(Diagnostic)]
#[diag("not a function")]
pub(crate) struct MustImplementNotFunction {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub span_note: MustImplementNotFunctionSpanNote,
    #[subdiagnostic]
    pub note: MustImplementNotFunctionNote,
}

#[derive(Subdiagnostic)]
#[note("required by this annotation")]
pub(crate) struct MustImplementNotFunctionSpanNote {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[note("all `#[rustc_must_implement_one_of]` arguments must be associated function names")]
pub(crate) struct MustImplementNotFunctionNote {}

#[derive(Diagnostic)]
#[diag("function not found in this trait")]
pub(crate) struct FunctionNotFoundInTrait {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("functions names are duplicated")]
#[note("all `#[rustc_must_implement_one_of]` arguments must be unique")]
pub(crate) struct FunctionNamesDuplicated {
    #[primary_span]
    pub spans: Vec<Span>,
}

#[derive(Diagnostic)]
#[diag("there is no parameter `{$argument_name}` on trait `{$trait_name}`")]
pub(crate) struct UnknownFormatParameterForOnUnimplementedAttr {
    pub argument_name: Symbol,
    pub trait_name: Ident,
    // `false` if we're in rustc_on_unimplemented, since its syntax is a lot more complex.
    #[help(r#"expect either a generic argument name or {"`{Self}`"} as format argument"#)]
    pub help: bool,
}
