//! Contains the data structures used by the diagnostic attribute family.

use rustc_span::{DesugaringKind, Span, Symbol};

/// Represents a format string in a on_unimplemented attribute,
/// like the "content" in `#[diagnostic::on_unimplemented(message = "content")]`
#[derive(Clone, Debug)]
pub struct OnUnimplementedFormatString {
    /// Symbol of the format string, i.e. `"content"`
    pub symbol: Symbol,
    /// The span of the format string, i.e. `"content"`
    pub span: Span,
    pub is_diagnostic_namespace_variant: bool,
}

#[derive(Debug)]
pub struct OnUnimplementedDirective {
    pub condition: Option<OnUnimplementedCondition>,
    pub subcommands: Vec<OnUnimplementedDirective>,
    pub message: Option<(Span, OnUnimplementedFormatString)>,
    pub label: Option<(Span, OnUnimplementedFormatString)>,
    pub notes: Vec<OnUnimplementedFormatString>,
    pub parent_label: Option<OnUnimplementedFormatString>,
    pub append_const_msg: Option<AppendConstMessage>,
}

/// For the `#[rustc_on_unimplemented]` attribute
#[derive(Default, Debug)]
pub struct OnUnimplementedNote {
    pub message: Option<String>,
    pub label: Option<String>,
    pub notes: Vec<String>,
    pub parent_label: Option<String>,
    // If none, should fall back to a generic message
    pub append_const_msg: Option<AppendConstMessage>,
}

/// Append a message for `[const] Trait` errors.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum AppendConstMessage {
    #[default]
    Default,
    Custom(Symbol, Span),
}

/// Like [std::fmt::Arguments] this is a string that has been parsed into "pieces",
/// either as string pieces or dynamic arguments.
#[derive(Debug)]
pub struct FormatString {
    pub input: Symbol,
    pub span: Span,
    pub pieces: Vec<Piece>,
    /// The formatting string was parsed successfully but with warnings
    pub warnings: Vec<FormatWarning>,
}

#[derive(Debug)]
pub enum Piece {
    Lit(String),
    Arg(FormatArg),
}

#[derive(Debug)]
pub enum FormatArg {
    // A generic parameter, like `{T}` if we're on the `From<T>` trait.
    GenericParam {
        generic_param: Symbol,
    },
    // `{Self}`
    SelfUpper,
    /// `{This}` or `{TraitName}`
    This,
    /// The sugared form of the trait
    Trait,
    /// what we're in, like a function, method, closure etc.
    ItemContext,
    /// What the user typed, if it doesn't match anything we can use.
    AsIs(String),
}

#[derive(Debug)]
pub enum FormatWarning {
    UnknownParam { argument_name: Symbol, span: Span },
    PositionalArgument { span: Span, help: String },
    InvalidSpecifier { name: String, span: Span },
    FutureIncompat { span: Span, help: String },
}

/// Represents the `on` filter in `#[rustc_on_unimplemented]`.
#[derive(Debug)]
pub struct OnUnimplementedCondition {
    pub span: Span,
    pub pred: Predicate,
}

/// Predicate(s) in `#[rustc_on_unimplemented]`'s `on` filter. See [`OnUnimplementedCondition`].
///
/// It is similar to the predicate in the `cfg` attribute,
/// and may contain nested predicates.
#[derive(Debug)]
pub enum Predicate {
    /// A condition like `on(crate_local)`.
    Flag(Flag),
    /// A match, like `on(Rhs = "Whatever")`.
    Match(NameValue),
    /// Negation, like `on(not($pred))`.
    Not(Box<Predicate>),
    /// True if all predicates are true, like `on(all($a, $b, $c))`.
    All(Vec<Predicate>),
    /// True if any predicate is true, like `on(any($a, $b, $c))`.
    Any(Vec<Predicate>),
}

impl Predicate {
    pub fn eval(&self, eval: &mut impl FnMut(FlagOrNv<'_>) -> bool) -> bool {
        match self {
            Predicate::Flag(flag) => eval(FlagOrNv::Flag(flag)),
            Predicate::Match(nv) => eval(FlagOrNv::NameValue(nv)),
            Predicate::Not(not) => !not.eval(eval),
            Predicate::All(preds) => preds.into_iter().all(|pred| pred.eval(eval)),
            Predicate::Any(preds) => preds.into_iter().any(|pred| pred.eval(eval)),
        }
    }
}

/// Represents a `MetaWord` in an `on`-filter.
#[derive(Debug, Clone, Copy)]
pub enum Flag {
    /// Whether the code causing the trait bound to not be fulfilled
    /// is part of the user's crate.
    CrateLocal,
    /// Whether the obligation is user-specified rather than derived.
    Direct,
    /// Whether we are in some kind of desugaring like
    /// `?` or `try { .. }`.
    FromDesugaring,
}

/// A `MetaNameValueStr` in an `on`-filter.
///
/// For example, `#[rustc_on_unimplemented(on(name = "value", message = "hello"))]`.
#[derive(Debug, Clone)]
pub struct NameValue {
    pub name: Name,
    /// Something like `"&str"` or `"alloc::string::String"`,
    /// in which case it just contains a single string piece.
    /// But if it is something like `"&[{A}]"` then it must be formatted later.
    pub value: FilterFormatString,
}

/// The valid names of the `on` filter.
#[derive(Debug, Clone, Copy)]
pub enum Name {
    Cause,
    FromDesugaring,
    SelfUpper,
    GenericArg(Symbol),
}

#[derive(Debug, Clone)]
pub enum FlagOrNv<'p> {
    Flag(&'p Flag),
    NameValue(&'p NameValue),
}

/// Represents a value inside an `on` filter.
///
/// For example, `#[rustc_on_unimplemented(on(name = "value", message = "hello"))]`.
/// If it is a simple literal like this then `pieces` will be `[LitOrArg::Lit("value")]`.
/// The `Arg` variant is used when it contains formatting like
/// `#[rustc_on_unimplemented(on(Self = "&[{A}]", message = "hello"))]`.
#[derive(Debug, Clone)]
pub struct FilterFormatString {
    pub pieces: Vec<LitOrArg>,
}

#[derive(Debug, Clone)]
pub enum LitOrArg {
    Lit(String),
    Arg(String),
}

/// Used with `OnUnimplementedCondition::matches_predicate` to evaluate the
/// [`OnUnimplementedCondition`].
///
/// For example, given a
/// ```rust,ignore (just an example)
/// #[rustc_on_unimplemented(
///     on(all(from_desugaring = "QuestionMark"),
///         message = "the `?` operator can only be used in {ItemContext} \
///                     that returns `Result` or `Option` \
///                     (or another type that implements `{FromResidual}`)",
///         label = "cannot use the `?` operator in {ItemContext} that returns `{Self}`",
///         parent_label = "this function should return `Result` or `Option` to accept `?`"
///     ),
/// )]
/// pub trait FromResidual<R = <Self as Try>::Residual> {
///    ...
/// }
///
/// async fn an_async_function() -> u32 {
///     let x: Option<u32> = None;
///     x?; //~ ERROR the `?` operator
///     22
/// }
///  ```
/// it will look like this:
///
/// ```rust,ignore (just an example)
/// ConditionOptions {
///     self_types: ["u32", "{integral}"],
///     from_desugaring: Some("QuestionMark"),
///     cause: None,
///     crate_local: false,
///     direct: true,
///     generic_args: [("Self","u32"),
///         ("R", "core::option::Option<core::convert::Infallible>"),
///         ("R", "core::option::Option<T>" ),
///     ],
/// }
/// ```
#[derive(Debug)]
pub struct ConditionOptions {
    /// All the self types that may apply.
    pub self_types: Vec<String>,
    // The kind of compiler desugaring.
    pub from_desugaring: Option<DesugaringKind>,
    /// Match on a variant of [rustc_infer::traits::ObligationCauseCode].
    pub cause: Option<String>,
    pub crate_local: bool,
    /// Is the obligation "directly" user-specified, rather than derived?
    pub direct: bool,
    // A list of the generic arguments and their reified types.
    pub generic_args: Vec<(Symbol, String)>,
}

impl ConditionOptions {
    pub fn has_flag(&self, name: Flag) -> bool {
        match name {
            Flag::CrateLocal => self.crate_local,
            Flag::Direct => self.direct,
            Flag::FromDesugaring => self.from_desugaring.is_some(),
        }
    }
    pub fn contains(&self, name: Name, value: String) -> bool {
        match name {
            Name::SelfUpper => self.self_types.contains(&value),
            Name::FromDesugaring => self.from_desugaring.is_some_and(|ds| ds.matches(&value)),
            Name::Cause => self.cause == Some(value),
            Name::GenericArg(arg) => self.generic_args.contains(&(arg, value)),
        }
    }
}
