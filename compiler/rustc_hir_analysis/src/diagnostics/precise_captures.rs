use rustc_errors::E0799;
use rustc_macros::Diagnostic;
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[diag("`impl Trait` must mention all {$kind} parameters in scope in `use<...>`")]
#[note(
    "currently, all {$kind} parameters are required to be mentioned in the precise captures list"
)]
pub(crate) struct ParamNotCaptured {
    #[primary_span]
    pub opaque_span: Span,
    #[label("{$kind} parameter is implicitly captured by this `impl Trait`")]
    pub param_span: Span,
    pub kind: &'static str,
}

#[derive(Diagnostic)]
#[diag("`impl Trait` must mention the `Self` type of the trait in `use<...>`")]
#[note("currently, all type parameters are required to be mentioned in the precise captures list")]
pub(crate) struct SelfTyNotCaptured {
    #[primary_span]
    pub opaque_span: Span,
    #[label("`Self` type parameter is implicitly captured by this `impl Trait`")]
    pub trait_span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "`impl Trait` captures lifetime parameter, but it is not mentioned in `use<...>` precise captures list"
)]
pub(crate) struct LifetimeNotCaptured {
    #[primary_span]
    pub use_span: Span,
    #[label("this lifetime parameter is captured")]
    pub param_span: Span,
    #[label("lifetime captured due to being mentioned in the bounds of the `impl Trait`")]
    pub opaque_span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "`impl Trait` captures lifetime parameter, but it is not mentioned in `use<...>` precise captures list"
)]
pub(crate) struct LifetimeImplicitlyCaptured {
    #[primary_span]
    pub opaque_span: Span,
    #[label("all lifetime parameters originating from a trait are captured implicitly")]
    pub param_span: Span,
}

#[derive(Diagnostic)]
#[diag("expected {$kind} parameter in `use<...>` precise captures list, found {$found}")]
pub(crate) struct BadPreciseCapture {
    #[primary_span]
    pub span: Span,
    pub kind: &'static str,
    pub found: String,
}

#[derive(Diagnostic)]
#[diag("`Self` can't be captured in `use<...>` precise captures list, since it is an alias", code = E0799)]
pub(crate) struct PreciseCaptureSelfAlias {
    #[primary_span]
    pub span: Span,
    #[label("`Self` is not a generic argument, but an alias to the type of the {$what}")]
    pub self_span: Span,
    pub what: &'static str,
}

#[derive(Diagnostic)]
#[diag("cannot capture parameter `{$name}` twice")]
pub(crate) struct DuplicatePreciseCapture {
    #[primary_span]
    pub first_span: Span,
    pub name: Symbol,
    #[label("parameter captured again here")]
    pub second_span: Span,
}

#[derive(Diagnostic)]
#[diag("lifetime parameter `{$name}` must be listed before non-lifetime parameters")]
pub(crate) struct LifetimesMustBeFirst {
    #[primary_span]
    pub lifetime_span: Span,
    pub name: Symbol,
    #[label("move the lifetime before this parameter")]
    pub other_span: Span,
}
