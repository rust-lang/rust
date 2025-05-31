use rustc_ast::{MetaItemInner, MetaItemKind, MetaItemLit};
use rustc_parse_format::{ParseMode, Parser, Piece, Position};
use rustc_span::{DesugaringKind, Ident, Span, Symbol, kw, sym};

use crate::errors::InvalidOnClause;

/// Represents the `on` filter in `#[rustc_on_unimplemented]`.
#[derive(Debug)]
pub(crate) struct OnUnimplementedCondition {
    span: Span,
    pred: Predicate,
}

impl OnUnimplementedCondition {
    pub(crate) fn span(&self) -> Span {
        self.span
    }

    pub(crate) fn matches_predicate(&self, options: &ConditionOptions) -> bool {
        self.pred.eval(&mut |p| match p {
            FlagOrNv::Flag(b) => options.has_flag(*b),
            FlagOrNv::NameValue(NameValue { name, value }) => {
                let value = value.format(&options.generic_args);
                options.contains(*name, value)
            }
        })
    }

    pub(crate) fn parse(
        input: &MetaItemInner,
        generics: &[Symbol],
    ) -> Result<Self, InvalidOnClause> {
        let span = input.span();
        let pred = Predicate::parse(input, generics)?;
        Ok(OnUnimplementedCondition { span, pred })
    }
}

/// Predicate(s) in `#[rustc_on_unimplemented]`'s `on` filter. See [`OnUnimplementedCondition`].
///
/// It is similar to the predicate in the `cfg` attribute,
/// and may contain nested predicates.
#[derive(Debug)]
enum Predicate {
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
    fn parse(input: &MetaItemInner, generics: &[Symbol]) -> Result<Self, InvalidOnClause> {
        let meta_item = match input {
            MetaItemInner::MetaItem(meta_item) => meta_item,
            MetaItemInner::Lit(lit) => {
                return Err(InvalidOnClause::UnsupportedLiteral { span: lit.span });
            }
        };

        let Some(predicate) = meta_item.ident() else {
            return Err(InvalidOnClause::ExpectedIdentifier {
                span: meta_item.path.span,
                path: meta_item.path.clone(),
            });
        };

        match meta_item.kind {
            MetaItemKind::List(ref mis) => match predicate.name {
                sym::any => Ok(Predicate::Any(Predicate::parse_sequence(mis, generics)?)),
                sym::all => Ok(Predicate::All(Predicate::parse_sequence(mis, generics)?)),
                sym::not => match &**mis {
                    [one] => Ok(Predicate::Not(Box::new(Predicate::parse(one, generics)?))),
                    [first, .., last] => Err(InvalidOnClause::ExpectedOnePredInNot {
                        span: first.span().to(last.span()),
                    }),
                    [] => Err(InvalidOnClause::ExpectedOnePredInNot { span: meta_item.span }),
                },
                invalid_pred => {
                    Err(InvalidOnClause::InvalidPredicate { span: predicate.span, invalid_pred })
                }
            },
            MetaItemKind::NameValue(MetaItemLit { symbol, .. }) => {
                let name = Name::parse(predicate, generics)?;
                let value = FilterFormatString::parse(symbol);
                let kv = NameValue { name, value };
                Ok(Predicate::Match(kv))
            }
            MetaItemKind::Word => {
                let flag = Flag::parse(predicate)?;
                Ok(Predicate::Flag(flag))
            }
        }
    }

    fn parse_sequence(
        sequence: &[MetaItemInner],
        generics: &[Symbol],
    ) -> Result<Vec<Self>, InvalidOnClause> {
        sequence.iter().map(|item| Predicate::parse(item, generics)).collect()
    }

    fn eval(&self, eval: &mut impl FnMut(FlagOrNv<'_>) -> bool) -> bool {
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
enum Flag {
    /// Whether the code causing the trait bound to not be fulfilled
    /// is part of the user's crate.
    CrateLocal,
    /// Whether the obligation is user-specified rather than derived.
    Direct,
    /// Whether we are in some kind of desugaring like
    /// `?` or `try { .. }`.
    FromDesugaring,
}

impl Flag {
    fn parse(Ident { name, span }: Ident) -> Result<Self, InvalidOnClause> {
        match name {
            sym::crate_local => Ok(Flag::CrateLocal),
            sym::direct => Ok(Flag::Direct),
            sym::from_desugaring => Ok(Flag::FromDesugaring),
            invalid_flag => Err(InvalidOnClause::InvalidFlag { invalid_flag, span }),
        }
    }
}

/// A `MetaNameValueStr` in an `on`-filter.
///
/// For example, `#[rustc_on_unimplemented(on(name = "value", message = "hello"))]`.
#[derive(Debug, Clone)]
struct NameValue {
    name: Name,
    /// Something like `"&str"` or `"alloc::string::String"`,
    /// in which case it just contains a single string piece.
    /// But if it is something like `"&[{A}]"` then it must be formatted later.
    value: FilterFormatString,
}

/// The valid names of the `on` filter.
#[derive(Debug, Clone, Copy)]
enum Name {
    Cause,
    FromDesugaring,
    SelfUpper,
    GenericArg(Symbol),
}

impl Name {
    fn parse(Ident { name, span }: Ident, generics: &[Symbol]) -> Result<Self, InvalidOnClause> {
        match name {
            kw::SelfUpper => Ok(Name::SelfUpper),
            sym::from_desugaring => Ok(Name::FromDesugaring),
            sym::cause => Ok(Name::Cause),
            generic if generics.contains(&generic) => Ok(Name::GenericArg(generic)),
            invalid_name => Err(InvalidOnClause::InvalidName { invalid_name, span }),
        }
    }
}

#[derive(Debug, Clone)]
enum FlagOrNv<'p> {
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
struct FilterFormatString {
    pieces: Vec<LitOrArg>,
}

#[derive(Debug, Clone)]
enum LitOrArg {
    Lit(String),
    Arg(String),
}

impl FilterFormatString {
    fn parse(input: Symbol) -> Self {
        let pieces = Parser::new(input.as_str(), None, None, false, ParseMode::Diagnostic)
            .map(|p| match p {
                Piece::Lit(s) => LitOrArg::Lit(s.to_owned()),
                // We just ignore formatspecs here
                Piece::NextArgument(a) => match a.position {
                    // In `TypeErrCtxt::on_unimplemented_note` we substitute `"{integral}"` even
                    // if the integer type has been resolved, to allow targeting all integers.
                    // `"{integer}"` and `"{float}"` come from numerics that haven't been inferred yet,
                    // from the `Display` impl of `InferTy` to be precise.
                    //
                    // Don't try to format these later!
                    Position::ArgumentNamed(arg @ "integer" | arg @ "integral" | arg @ "float") => {
                        LitOrArg::Lit(format!("{{{arg}}}"))
                    }

                    // FIXME(mejrs) We should check if these correspond to a generic of the trait.
                    Position::ArgumentNamed(arg) => LitOrArg::Arg(arg.to_owned()),

                    // FIXME(mejrs) These should really be warnings/errors
                    Position::ArgumentImplicitlyIs(_) => LitOrArg::Lit(String::from("{}")),
                    Position::ArgumentIs(idx) => LitOrArg::Lit(format!("{{{idx}}}")),
                },
            })
            .collect();
        Self { pieces }
    }

    fn format(&self, generic_args: &[(Symbol, String)]) -> String {
        let mut ret = String::new();

        for piece in &self.pieces {
            match piece {
                LitOrArg::Lit(s) => ret.push_str(s),
                LitOrArg::Arg(arg) => {
                    let s = Symbol::intern(arg);
                    match generic_args.iter().find(|(k, _)| *k == s) {
                        Some((_, val)) => ret.push_str(val),
                        None => {
                            // FIXME(mejrs) If we start checking as mentioned in
                            // FilterFormatString::parse then this shouldn't happen
                            let _ = std::fmt::write(&mut ret, format_args!("{{{s}}}"));
                        }
                    }
                }
            }
        }

        ret
    }
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
pub(crate) struct ConditionOptions {
    /// All the self types that may apply.
    pub(crate) self_types: Vec<String>,
    // The kind of compiler desugaring.
    pub(crate) from_desugaring: Option<DesugaringKind>,
    /// Match on a variant of [rustc_infer::traits::ObligationCauseCode].
    pub(crate) cause: Option<String>,
    pub(crate) crate_local: bool,
    /// Is the obligation "directly" user-specified, rather than derived?
    pub(crate) direct: bool,
    // A list of the generic arguments and their reified types.
    pub(crate) generic_args: Vec<(Symbol, String)>,
}

impl ConditionOptions {
    fn has_flag(&self, name: Flag) -> bool {
        match name {
            Flag::CrateLocal => self.crate_local,
            Flag::Direct => self.direct,
            Flag::FromDesugaring => self.from_desugaring.is_some(),
        }
    }
    fn contains(&self, name: Name, value: String) -> bool {
        match name {
            Name::SelfUpper => self.self_types.contains(&value),
            Name::FromDesugaring => self.from_desugaring.is_some_and(|ds| ds.matches(&value)),
            Name::Cause => self.cause == Some(value),
            Name::GenericArg(arg) => self.generic_args.contains(&(arg, value)),
        }
    }
}
