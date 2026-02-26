//! Contains the data structures used by the diagnostic attribute family.
use std::fmt;
use std::fmt::Debug;

pub use rustc_ast::attr::data_structures::*;
use rustc_macros::{Decodable, Encodable, HashStable_Generic, PrintAttribute};
use rustc_span::{DesugaringKind, Span, Symbol, kw};
use thin_vec::ThinVec;
use tracing::{debug, info};

use crate::attrs::PrintAttribute;

#[derive(Clone, Default, Debug, HashStable_Generic, Encodable, Decodable, PrintAttribute)]
pub struct Directive {
    pub is_rustc_attr: bool,
    pub condition: Option<OnUnimplementedCondition>,
    pub subcommands: ThinVec<Directive>,
    pub message: Option<(Span, FormatString)>,
    pub label: Option<(Span, FormatString)>,
    pub notes: ThinVec<FormatString>,
    pub parent_label: Option<FormatString>,
    pub append_const_msg: Option<AppendConstMessage>,
}

impl Directive {
    /// Visit all the generic arguments used in the attribute, to see whether they are actually a
    /// generic of the item. If not then `visit` must issue a diagnostic.
    ///
    /// We can't check this while parsing the attribute because `rustc_attr_parsing` doesn't have
    /// access to the item an attribute is on. Instead we later call this function in `check_attr`.
    pub fn visit_params(&self, visit: &mut impl FnMut(Symbol, Span)) {
        if let Some(condition) = &self.condition {
            condition.visit_params(visit);
        }

        for subcommand in &self.subcommands {
            subcommand.visit_params(visit);
        }

        if let Some((_, message)) = &self.message {
            message.visit_params(visit);
        }
        if let Some((_, label)) = &self.label {
            label.visit_params(visit);
        }

        for note in &self.notes {
            note.visit_params(visit);
        }

        if let Some(parent_label) = &self.parent_label {
            parent_label.visit_params(visit);
        }
    }

    pub fn evaluate_directive(
        &self,
        trait_name: impl Debug,
        condition_options: &ConditionOptions,
        args: &FormatArgs,
    ) -> OnUnimplementedNote {
        let mut message = None;
        let mut label = None;
        let mut notes = Vec::new();
        let mut parent_label = None;
        let mut append_const_msg = None;
        info!(
            "evaluate_directive({:?}, trait_ref={:?}, options={:?}, args ={:?})",
            self, trait_name, condition_options, args
        );

        for command in self.subcommands.iter().chain(Some(self)).rev() {
            debug!(?command);
            if let Some(ref condition) = command.condition
                && !condition.matches_predicate(condition_options)
            {
                debug!("evaluate_directive: skipping {:?} due to condition", command);
                continue;
            }
            debug!("evaluate_directive: {:?} succeeded", command);
            if let Some(ref message_) = command.message {
                message = Some(message_.clone());
            }

            if let Some(ref label_) = command.label {
                label = Some(label_.clone());
            }

            notes.extend(command.notes.clone());

            if let Some(ref parent_label_) = command.parent_label {
                parent_label = Some(parent_label_.clone());
            }

            append_const_msg = command.append_const_msg;
        }

        OnUnimplementedNote {
            label: label.map(|l| l.1.format(args)),
            message: message.map(|m| m.1.format(args)),
            notes: notes.into_iter().map(|n| n.format(args)).collect(),
            parent_label: parent_label.map(|e_s| e_s.format(args)),
            append_const_msg,
        }
    }
}

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
#[derive(HashStable_Generic, Encodable, Decodable, PrintAttribute)]
pub enum AppendConstMessage {
    #[default]
    Default,
    Custom(Symbol, Span),
}

/// Like [std::fmt::Arguments] this is a string that has been parsed into "pieces",
/// either as string pieces or dynamic arguments.
#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable, PrintAttribute)]
pub struct FormatString {
    pub input: Symbol,
    pub span: Span,
    pub pieces: ThinVec<Piece>,
}
impl FormatString {
    pub fn format(&self, args: &FormatArgs) -> String {
        let mut ret = String::new();
        for piece in &self.pieces {
            match piece {
                Piece::Lit(s) | Piece::Arg(FormatArg::AsIs(s)) => ret.push_str(s.as_str()),

                // `A` if we have `trait Trait<A> {}` and `note = "i'm the actual type of {A}"`
                Piece::Arg(FormatArg::GenericParam { generic_param, .. }) => {
                    match args.generic_args.iter().find(|(p, _)| p == generic_param) {
                        Some((_, val)) => ret.push_str(val.as_str()),

                        None => {
                            // Apparently this was not actually a generic parameter, so lets write
                            // what the user wrote.
                            let _ = fmt::write(&mut ret, format_args!("{{{generic_param}}}"));
                        }
                    }
                }
                // `{Self}`
                Piece::Arg(FormatArg::SelfUpper) => {
                    let slf = match args.generic_args.iter().find(|(p, _)| *p == kw::SelfUpper) {
                        Some((_, val)) => val.to_string(),
                        None => "Self".to_string(),
                    };
                    ret.push_str(&slf);
                }

                // It's only `rustc_onunimplemented` from here
                Piece::Arg(FormatArg::This) => ret.push_str(&args.this),
                Piece::Arg(FormatArg::Trait) => {
                    let _ = fmt::write(&mut ret, format_args!("{}", &args.trait_sugared));
                }
                Piece::Arg(FormatArg::ItemContext) => ret.push_str(args.item_context),
            }
        }
        ret
    }

    fn visit_params(&self, visit: &mut impl FnMut(Symbol, Span)) {
        for piece in &self.pieces {
            if let Piece::Arg(FormatArg::GenericParam { generic_param, span }) = piece {
                visit(*generic_param, *span);
            }
        }
    }
}

/// Arguments to fill a [FormatString] with.
///
/// For example, given a
/// ```rust,ignore (just an example)
///
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
/// FormatArgs {
///     this: "FromResidual",
///     trait_sugared: "FromResidual<Option<Infallible>>",
///     item_context: "an async function",
///     generic_args: [("Self", "u32"), ("R", "Option<Infallible>")],
/// }
/// ```
#[derive(Debug)]
pub struct FormatArgs {
    pub this: String,
    pub trait_sugared: String,
    pub item_context: &'static str,
    pub generic_args: Vec<(Symbol, String)>,
}

#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable, PrintAttribute)]
pub enum Piece {
    Lit(Symbol),
    Arg(FormatArg),
}

#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable, PrintAttribute)]
pub enum FormatArg {
    // A generic parameter, like `{T}` if we're on the `From<T>` trait.
    GenericParam {
        generic_param: Symbol,
        span: Span,
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
    AsIs(Symbol),
}

/// Represents the `on` filter in `#[rustc_on_unimplemented]`.
#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable, PrintAttribute)]
pub struct OnUnimplementedCondition {
    pub span: Span,
    pub pred: Predicate,
}
impl OnUnimplementedCondition {
    pub fn matches_predicate(self: &OnUnimplementedCondition, options: &ConditionOptions) -> bool {
        self.pred.eval(&mut |p| match p {
            FlagOrNv::Flag(b) => options.has_flag(*b),
            FlagOrNv::NameValue(NameValue { name, value }) => {
                let value = value.format(&options.generic_args);
                options.contains(*name, value)
            }
        })
    }

    pub fn visit_params(&self, visit: &mut impl FnMut(Symbol, Span)) {
        self.pred.visit_params(self.span, visit);
    }
}

/// Predicate(s) in `#[rustc_on_unimplemented]`'s `on` filter. See [`OnUnimplementedCondition`].
///
/// It is similar to the predicate in the `cfg` attribute,
/// and may contain nested predicates.
#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable, PrintAttribute)]
pub enum Predicate {
    /// A condition like `on(crate_local)`.
    Flag(Flag),
    /// A match, like `on(Rhs = "Whatever")`.
    Match(NameValue),
    /// Negation, like `on(not($pred))`.
    Not(Box<Predicate>),
    /// True if all predicates are true, like `on(all($a, $b, $c))`.
    All(ThinVec<Predicate>),
    /// True if any predicate is true, like `on(any($a, $b, $c))`.
    Any(ThinVec<Predicate>),
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

    pub fn visit_params(&self, span: Span, visit: &mut impl FnMut(Symbol, Span)) {
        match self {
            Predicate::Flag(_) => {}
            Predicate::Match(nv) => nv.visit_params(span, visit),
            Predicate::Not(not) => not.visit_params(span, visit),
            Predicate::All(preds) | Predicate::Any(preds) => {
                preds.iter().for_each(|pred| pred.visit_params(span, visit))
            }
        }
    }
}

/// Represents a `MetaWord` in an `on`-filter.
#[derive(Clone, Copy, Debug, HashStable_Generic, Encodable, Decodable, PrintAttribute)]
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
#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable, PrintAttribute)]
pub struct NameValue {
    pub name: Name,
    /// Something like `"&str"` or `"alloc::string::String"`,
    /// in which case it just contains a single string piece.
    /// But if it is something like `"&[{A}]"` then it must be formatted later.
    pub value: FilterFormatString,
}

impl NameValue {
    pub fn visit_params(&self, span: Span, visit: &mut impl FnMut(Symbol, Span)) {
        if let Name::GenericArg(arg) = self.name {
            visit(arg, span);
        }
        self.value.visit_params(span, visit);
    }
}

/// The valid names of the `on` filter.
#[derive(Clone, Copy, Debug, HashStable_Generic, Encodable, Decodable, PrintAttribute)]
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
#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable, PrintAttribute)]
pub struct FilterFormatString {
    pub pieces: ThinVec<LitOrArg>,
}

impl FilterFormatString {
    fn format(&self, generic_args: &[(Symbol, String)]) -> String {
        let mut ret = String::new();

        for piece in &self.pieces {
            match piece {
                LitOrArg::Lit(s) => ret.push_str(s.as_str()),
                LitOrArg::Arg(s) => match generic_args.iter().find(|(k, _)| k == s) {
                    Some((_, val)) => ret.push_str(val),
                    None => {
                        let _ = std::fmt::write(&mut ret, format_args!("{{{s}}}"));
                    }
                },
            }
        }

        ret
    }
    pub fn visit_params(&self, span: Span, visit: &mut impl FnMut(Symbol, Span)) {
        for piece in &self.pieces {
            if let LitOrArg::Arg(arg) = piece {
                visit(*arg, span);
            }
        }
    }
}

#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable, PrintAttribute)]
pub enum LitOrArg {
    Lit(Symbol),
    Arg(Symbol),
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
    /// Match on a variant of rustc_infer's `ObligationCauseCode`.
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
