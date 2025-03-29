use rustc_ast::MetaItemInner;
use rustc_attr_parsing as attr;
use rustc_middle::ty::{self, TyCtxt};
use rustc_parse_format::{ParseMode, Parser, Piece, Position};
use rustc_span::{DesugaringKind, Span, Symbol, kw, sym};

/// A predicate in an attribute using on, all, any,
/// similar to a cfg predicate.
#[derive(Debug)]
pub struct Condition {
    pub inner: MetaItemInner,
}

impl Condition {
    pub fn span(&self) -> Span {
        self.inner.span()
    }

    pub fn matches_predicate<'tcx>(&self, tcx: TyCtxt<'tcx>, options: &ConditionOptions) -> bool {
        attr::eval_condition(&self.inner, tcx.sess, Some(tcx.features()), &mut |cfg| {
            let value = cfg.value.map(|v| {
                // `with_no_visible_paths` is also used when generating the options,
                // so we need to match it here.
                ty::print::with_no_visible_paths!({
                    Parser::new(v.as_str(), None, None, false, ParseMode::Format)
                        .map(|p| match p {
                            Piece::Lit(s) => s.to_owned(),
                            Piece::NextArgument(a) => match a.position {
                                Position::ArgumentNamed(arg) => {
                                    let s = Symbol::intern(arg);
                                    match options.generic_args.iter().find(|(k, _)| *k == s) {
                                        Some((_, val)) => val.to_string(),
                                        None => format!("{{{arg}}}"),
                                    }
                                }
                                Position::ArgumentImplicitlyIs(_) => String::from("{}"),
                                Position::ArgumentIs(idx) => format!("{{{idx}}}"),
                            },
                        })
                        .collect()
                })
            });

            options.contains(cfg.name, &value)
        })
    }
}

/// Used with `Condition::matches_predicate` to test whether the condition applies
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
    /// for example
    pub self_types: Vec<String>,
    // The kind of compiler desugaring.
    pub from_desugaring: Option<DesugaringKind>,
    /// Match on a variant of [rustc_infer::traits::ObligationCauseCode]
    pub cause: Option<String>,
    pub crate_local: bool,
    /// Is the obligation "directly" user-specified, rather than derived?
    pub direct: bool,
    // A list of the generic arguments and their reified types
    pub generic_args: Vec<(Symbol, String)>,
}

impl ConditionOptions {
    pub fn contains(&self, key: Symbol, value: &Option<String>) -> bool {
        match (key, value) {
            (sym::_Self | kw::SelfUpper, Some(value)) => self.self_types.contains(&value),
            // from_desugaring as a flag
            (sym::from_desugaring, None) => self.from_desugaring.is_some(),
            // from_desugaring as key == value
            (sym::from_desugaring, Some(v)) if let Some(ds) = self.from_desugaring => ds.matches(v),
            (sym::cause, Some(value)) => self.cause.as_deref() == Some(value),
            (sym::crate_local, None) => self.crate_local,
            (sym::direct, None) => self.direct,
            (other, Some(value)) => {
                self.generic_args.iter().any(|(k, v)| *k == other && v == value)
            }
            _ => false,
        }
    }
}
