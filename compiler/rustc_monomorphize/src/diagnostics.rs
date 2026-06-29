use rustc_macros::Diagnostic;
use rustc_middle::ty::{Instance, Ty};
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[diag("reached the recursion limit while instantiating `{$instance}`")]
pub(crate) struct RecursionLimit<'tcx> {
    #[primary_span]
    pub span: Span,
    pub instance: Instance<'tcx>,
    #[note("`{$def_path_str}` defined here")]
    pub def_span: Span,
    pub def_path_str: String,
}

#[derive(Diagnostic)]
#[diag("missing optimized MIR for `{$instance}` in the crate `{$crate_name}`")]
pub(crate) struct NoOptimizedMir {
    #[note(
        "missing optimized MIR for this item (was the crate `{$crate_name}` compiled with `--emit=metadata`?)"
    )]
    pub span: Span,
    pub crate_name: Symbol,
    pub instance: String,
}

#[derive(Diagnostic)]
#[diag("moving {$size} bytes")]
#[note(
    "the current maximum size is {$limit}, but it can be customized with the move_size_limit attribute: `#![move_size_limit = \"...\"]`"
)]
pub(crate) struct LargeAssignmentsLint {
    #[label("value moved from here")]
    pub span: Span,
    pub size: u64,
    pub limit: u64,
}

#[derive(Diagnostic)]
#[diag("symbol `{$symbol}` is already defined")]
pub(crate) struct SymbolAlreadyDefined {
    #[primary_span]
    pub span: Option<Span>,
    pub symbol: String,
}

#[derive(Diagnostic)]
#[diag("unexpected error occurred while dumping monomorphization stats: {$error}")]
pub(crate) struct CouldntDumpMonoStats {
    pub error: String,
}

#[derive(Diagnostic)]
#[diag("the above error was encountered while instantiating `{$kind} {$instance}`")]
pub(crate) struct EncounteredErrorWhileInstantiating<'tcx> {
    #[primary_span]
    pub span: Span,
    pub kind: &'static str,
    pub instance: Instance<'tcx>,
}

#[derive(Diagnostic)]
#[diag("the above error was encountered while instantiating `global_asm`")]
pub(crate) struct EncounteredErrorWhileInstantiatingGlobalAsm {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("using `fn main` requires the standard library")]
#[help(
    "use `#![no_main]` to bypass the Rust generated entrypoint and declare a platform specific entrypoint yourself, usually with `#[no_mangle]`"
)]
pub(crate) struct StartNotFound;

#[derive(Diagnostic)]
#[diag("this function {$is_call ->
    [true] call
    *[false] definition
} uses {$is_scalable ->
    [true] scalable
    *[false] SIMD
} vector type `{$ty}` which (with the chosen ABI) requires the `{$required_feature}` target feature, which is not enabled{$is_call ->
    [true] {\" \"}in the caller
    *[false] {\"\"}
}")]
#[help(
    "consider enabling it globally (`-C target-feature=+{$required_feature}`) or locally (`#[target_feature(enable=\"{$required_feature}\")]`)"
)]
pub(crate) struct AbiErrorDisabledVectorType<'a> {
    #[primary_span]
    #[label(
        "function {$is_call ->
        [true] called
        *[false] defined
    } here"
    )]
    pub span: Span,
    pub required_feature: &'a str,
    pub ty: Ty<'a>,
    /// Whether this is a problem at a call site or at a declaration.
    pub is_call: bool,
    /// Whether this is a problem with a fixed length vector or a scalable vector
    pub is_scalable: bool,
}

#[derive(Diagnostic)]
#[diag(
    "this function {$is_call ->
        [true] call
        *[false] definition
    } uses unsized type `{$ty}` which is not supported with the chosen ABI"
)]
#[help("only rustic ABIs support unsized parameters")]
pub(crate) struct AbiErrorUnsupportedUnsizedParameter<'a> {
    #[primary_span]
    #[label(
        "function {$is_call ->
            [true] called
            *[false] defined
        } here"
    )]
    pub span: Span,
    pub ty: Ty<'a>,
    /// Whether this is a problem at a call site or at a declaration.
    pub is_call: bool,
}

#[derive(Diagnostic)]
#[diag(
    "this function {$is_call ->
        [true] call
        *[false] definition
    } uses SIMD vector type `{$ty}` which is not currently supported with the chosen ABI"
)]
pub(crate) struct AbiErrorUnsupportedVectorType<'a> {
    #[primary_span]
    #[label(
        "function {$is_call ->
            [true] called
            *[false] defined
        } here"
    )]
    pub span: Span,
    pub ty: Ty<'a>,
    /// Whether this is a problem at a call site or at a declaration.
    pub is_call: bool,
}

#[derive(Diagnostic)]
#[diag("this function {$is_call ->
    [true] call
    *[false] definition
} uses ABI \"{$abi}\" which requires the `{$required_feature}` target feature, which is not enabled{$is_call ->
    [true] {\" \"}in the caller
    *[false] {\"\"}
}")]
#[help(
    "consider enabling it globally (`-C target-feature=+{$required_feature}`) or locally (`#[target_feature(enable=\"{$required_feature}\")]`)"
)]
pub(crate) struct AbiRequiredTargetFeature<'a> {
    #[primary_span]
    #[label(
        "function {$is_call ->
            [true] called
            *[false] defined
        } here"
    )]
    pub span: Span,
    pub required_feature: &'a str,
    pub abi: &'a str,
    /// Whether this is a problem at a call site or at a declaration.
    pub is_call: bool,
}

#[derive(Diagnostic)]
#[diag("static initializer forms a cycle involving `{$head}`")]
#[note("cyclic static initializers are not supported for target `{$target}`")]
pub(crate) struct StaticInitializerCyclic<'a> {
    #[primary_span]
    pub span: Span,
    #[label("part of this cycle")]
    pub labels: Vec<Span>,
    pub head: &'a str,
    pub target: &'a str,
}
